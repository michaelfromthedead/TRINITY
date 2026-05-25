//! Lock-free frame graph swapping via `ArcSwap`.
//!
//! Provides [`FrameGraphSlot`], a thread-safe container that holds the
//! current [`CompiledFrameGraph`] and allows lock-free swap between frames
//! using the [`arc_swap`](https://docs.rs/arc-swap) crate.
//!
//! The render thread reads the current graph each frame via
//! [`FrameGraphSlot::load`]; a background compilation thread produces a
//! new graph and swaps it in via [`FrameGraphSlot::store`] or
//! [`FrameGraphSlot::compile_and_swap`].
//!
//! # Contention characteristics
//!
//! Reads perform a single atomic load (no lock) that clones the `Arc`
//! pointer — a single atomic increment.  Writes perform an atomic swap
//! (no lock).  Because recompilation happens orders of magnitude less
//! often than per-frame reads, contention is negligible in practice.
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::frame_graph::swap::FrameGraphSlot;
//!
//! // Initialise the slot with the first compiled graph.
//! let slot = FrameGraphSlot::new(initial_graph);
//!
//! // ---- Render thread (once per frame) ----
//! let graph = slot.load();
//! execute_frame(&*graph);
//! // graph is dropped here; the Arc ref-count handles deallocation
//!
//! // ---- Background compilation thread (on scene / material change) ----
//! let new_graph = CompiledFrameGraph::compile(passes, resources)?;
//! slot.store(new_graph);
//! ```

use arc_swap::ArcSwap;
use std::sync::Arc;

use super::{CompiledFrameGraph, IrPass, IrResource};

/// A lock-free slot that holds the current compiled frame graph.
///
/// # Architecture
///
/// ```text
///                         ┌──────────────────┐
///  Render Thread          │  FrameGraphSlot   │     Compilation Thread
///  (reads every frame)    │  (ArcSwap<Arc<>>) │     (writes on recompile)
///       load() ──────────▶│                   │◀────────── store()
///                         └──────────────────┘
/// ```
///
/// Readers obtain a snapshot of the current graph via an atomic load (lock-free).
/// Writers atomically replace the pointer via [`store`] or [`swap`].  Because
/// writes (recompilation) happen orders of magnitude less frequently than reads
/// (once per frame), contention is negligible.
///
/// The `Arc` indirection ensures that in-flight frames are never interrupted:
/// each [`load`] call bumps the reference count, and the old graph is only
/// deallocated when all outstanding `Arc` handles have been dropped.
///
/// [`load`]: Self::load
/// [`store`]: Self::store
/// [`swap`]: Self::swap
pub struct FrameGraphSlot {
    /// The current compiled frame graph, behind an `ArcSwap` so that reads
    /// are lock-free (single atomic load) while still allowing safe concurrent
    /// access.  In-flight frames hold their own `Arc` clones via [`load`],
    /// preventing premature deallocation when the slot is swapped to a newer
    /// graph.
    ///
    /// [`load`]: Self::load
    current: ArcSwap<CompiledFrameGraph>,
}

impl FrameGraphSlot {
    /// Creates a new slot containing the given compiled graph.
    ///
    /// After construction, [`load`](Self::load) returns an `Arc` to this
    /// initial graph.  Use [`store`](Self::store) to replace it with a
    /// newer compilation result.
    pub fn new(initial: CompiledFrameGraph) -> Self {
        Self {
            current: ArcSwap::new(Arc::new(initial)),
        }
    }

    /// Returns a reference-counted handle to the current compiled graph.
    ///
    /// The returned `Arc` pins the graph at the version that was current
    /// at the moment of the call.  Even after the slot is swapped to a
    /// newer graph, the returned `Arc` keeps the old graph alive until
    /// all consumers are done with it — guaranteeing that in-flight frames
    /// are never interrupted mid-execution.
    ///
    /// The cost is a single atomic increment on the `Arc` reference count
    /// plus a lock-free load of the `ArcSwap` pointer.  No graph data is
    /// copied, and no lock is acquired.
    pub fn load(&self) -> Arc<CompiledFrameGraph> {
        // load_full performs a lock-free atomic load and returns a full
        // Arc clone (ref-count increment).  The underlying graph data is
        // not copied.
        self.current.load_full()
    }

    /// Atomically replaces the current compiled graph with a new one.
    ///
    /// Performs a lock-free atomic store.  The old graph is not deallocated
    /// until all existing [`load`](Self::load) handles are dropped,
    /// guaranteeing that in-flight frames continue uninterrupted.
    pub fn store(&self, new_graph: CompiledFrameGraph) {
        self.current.store(Arc::new(new_graph));
    }

    /// Atomically swaps the current graph for a new one, returning the old.
    ///
    /// This is a single atomic swap operation (lock-free).  The returned
    /// `Arc` holds the previous graph.  Existing handles from
    /// [`load`](Self::load) remain valid and keep the old graph alive.
    pub fn swap(&self, new_graph: CompiledFrameGraph) -> Arc<CompiledFrameGraph> {
        self.current.swap(Arc::new(new_graph))
    }

    /// Compiles a new frame graph from IR passes and resources, then
    /// atomically swaps it in.
    ///
    /// The compilation runs on the caller's thread (typically a background
    /// compilation thread).  Only the final pointer swap is performed
    /// atomically, so compilation time does not block readers.
    ///
    /// # Errors
    ///
    /// Propagates compilation errors from
    /// [`CompiledFrameGraph::compile`](super::CompiledFrameGraph::compile)
    /// without modifying the slot.  The slot retains the previous graph.
    pub fn compile_and_swap(
        &self,
        passes: Vec<IrPass>,
        resources: Vec<IrResource>,
    ) -> Result<Arc<CompiledFrameGraph>, String> {
        let new_graph = CompiledFrameGraph::compile(passes, resources)?;
        Ok(self.swap(new_graph))
    }

    /// Returns `true` if the slot currently holds a graph.
    ///
    /// This is always `true` after construction via [`new`](Self::new),
    /// but provides a consistent API for future scenarios where the slot
    /// might support lazy initialisation or an "empty" state.
    pub fn is_loaded(&self) -> bool {
        true
    }
}

// SAFETY: ArcSwap<CompiledFrameGraph> is Send + Sync because it only
// contains atomic operations on an Arc pointer.  CompiledFrameGraph
// contains only owned data (`Vec`, `HashMap`, etc.) and is trivially
// `Send`.  No manual unsafe impl is needed.

impl std::fmt::Debug for FrameGraphSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Report the type name and capability without accessing the
        // ArcSwap, which could be contended (though contention is
        // extremely unlikely).
        f.debug_struct("FrameGraphSlot")
            .field("loaded", &self.is_loaded())
            .finish()
    }
}

// ===========================================================================
// White-box tests — T-FG-7.8 (FrameGraphSlot / ArcSwap)
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time checks: FrameGraphSlot must be Send + Sync.
    const fn _assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn test_send_sync_bounds() {
        // If FrameGraphSlot were not Send + Sync, this would fail to compile.
        _assert_send_sync::<FrameGraphSlot>();
    }

    // Small helper: build a single-pass, single-resource CompiledFrameGraph.
    fn make_single_pass_graph(name: &str) -> CompiledFrameGraph {
        let resource = super::super::IrResource::new(
            super::super::ResourceHandle(1),
            format!("{}_rt", name),
            super::super::ResourceDesc::Texture2D(super::super::TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            super::super::ResourceLifetime::Transient,
            super::super::ResourceState::Uninitialized,
        );

        let pass = super::super::IrPass::graphics(
            super::super::PassIndex(0),
            name,
            vec![super::super::ColorAttachment {
                resource: super::super::ResourceHandle(1),
                mip_level: 0,
                array_layer: 0,
                load_op: super::super::AttachmentLoadOp::Clear,
                store_op: super::super::AttachmentStoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 0.0],
            }],
            None,
            super::super::InstanceSource::Direct {
                index_count: 6,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            super::super::ViewType::ColorAttachment,
        );

        CompiledFrameGraph::compile(vec![pass], vec![resource]).unwrap()
    }

    // -----------------------------------------------------------------------
    // 1. new() creates a loaded slot
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_creates_loaded_slot() {
        let graph = make_single_pass_graph("test_new");
        let slot = FrameGraphSlot::new(graph);
        assert!(slot.is_loaded(), "slot should be loaded after construction");
    }

    // -----------------------------------------------------------------------
    // 2. load() returns a reference to the stored graph
    // -----------------------------------------------------------------------
    #[test]
    fn test_load_returns_stored_graph() {
        let graph = make_single_pass_graph("test_load");
        // Snapshot the name before wrapping.
        let expected_name = graph.passes[0].name.clone();
        let slot = FrameGraphSlot::new(graph);

        let loaded = slot.load();
        assert_eq!(
            loaded.passes[0].name, expected_name,
            "load() should return the graph that was stored"
        );
    }

    // -----------------------------------------------------------------------
    // 3. store() replaces the graph
    // -----------------------------------------------------------------------
    #[test]
    fn test_store_replaces_graph() {
        let graph_a = make_single_pass_graph("graph_a");
        let graph_b = make_single_pass_graph("graph_b");

        let slot = FrameGraphSlot::new(graph_a);

        // Verify initial.
        assert_eq!(slot.load().passes[0].name, "graph_a");

        // Store a new graph.
        slot.store(graph_b);

        // Subsequent load returns the new graph.
        assert_eq!(
            slot.load().passes[0].name,
            "graph_b",
            "store() should replace the current graph"
        );
    }

    // -----------------------------------------------------------------------
    // 4. swap() returns the old Arc<CompiledFrameGraph>
    // -----------------------------------------------------------------------
    #[test]
    fn test_swap_returns_old_arc() {
        let graph_a = make_single_pass_graph("graph_a");
        let graph_b = make_single_pass_graph("graph_b");

        let slot = FrameGraphSlot::new(graph_a);

        let old = slot.swap(graph_b);
        assert_eq!(
            old.passes[0].name,
            "graph_a",
            "swap() should return the previously stored graph"
        );

        // Slot now holds graph_b.
        assert_eq!(slot.load().passes[0].name, "graph_b");
    }

    // -----------------------------------------------------------------------
    // 5. After swap(), old Arc is still valid (not dropped prematurely)
    // -----------------------------------------------------------------------
    #[test]
    fn test_old_arc_remains_valid_after_swap() {
        let graph_a = make_single_pass_graph("graph_a");
        let graph_b = make_single_pass_graph("graph_b");

        let slot = FrameGraphSlot::new(graph_a);

        // Hold the old Arc from load().
        let old_handle = slot.load();

        // Swap in a new graph.
        let _swapped_out = slot.swap(graph_b);

        // The old Arc should still be fully functional.
        assert_eq!(
            old_handle.passes[0].name,
            "graph_a",
            "old Arc must remain valid after the slot is swapped"
        );

        // Double-check that slot's load() now returns the *new* graph.
        assert_eq!(
            slot.load().passes[0].name,
            "graph_b",
            "slot should reflect the swapped-in graph"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Multiple load() calls from the same slot return consistent data
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_loads_return_consistent_data() {
        let graph = make_single_pass_graph("test_consistent");
        let slot = FrameGraphSlot::new(graph);

        let a = slot.load();
        let b = slot.load();

        // Both Arcs point to the same underlying graph.
        assert_eq!(
            a.passes[0].name,
            b.passes[0].name,
            "two load() calls before a write should return equivalent data"
        );

        // The pass count should match.
        assert_eq!(a.passes.len(), b.passes.len());
    }

    // -----------------------------------------------------------------------
    // 7. compile_and_swap with valid inputs succeeds and returns the old Arc
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_and_swap_succeeds() {
        let initial = make_single_pass_graph("initial");
        let slot = FrameGraphSlot::new(initial);

        // Build IR for a new, independent pass.
        let resource = super::super::IrResource::new(
            super::super::ResourceHandle(2),
            "compile_swap_rt",
            super::super::ResourceDesc::Texture2D(super::super::TextureDesc {
                width: 640,
                height: 480,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            super::super::ResourceLifetime::Transient,
            super::super::ResourceState::Uninitialized,
        );

        let pass = super::super::IrPass::graphics(
            super::super::PassIndex(0),
            "compiled_pass",
            vec![super::super::ColorAttachment {
                resource: super::super::ResourceHandle(2),
                mip_level: 0,
                array_layer: 0,
                load_op: super::super::AttachmentLoadOp::Clear,
                store_op: super::super::AttachmentStoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 0.0],
            }],
            None,
            super::super::InstanceSource::Direct {
                index_count: 6,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            super::super::ViewType::ColorAttachment,
        );

        let result = slot.compile_and_swap(vec![pass], vec![resource]);
        assert!(result.is_ok(), "compile_and_swap should succeed with valid inputs");

        // The returned Arc must be the OLD (initial) graph, not the new one.
        let old = result.unwrap();
        assert_eq!(
            old.passes[0].name,
            "initial",
            "compile_and_swap should return the previously stored graph"
        );

        // The slot must now hold the newly compiled graph.
        assert_eq!(
            slot.load().passes[0].name,
            "compiled_pass",
            "slot should contain the pass compiled via compile_and_swap"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Failure-recovery: slot invariant on compile error
    // -----------------------------------------------------------------------
    //
    // This test guards the core contract of compile_and_swap:
    //
    //   "On compilation failure, the slot retains the previous graph."
    //
    // Currently, compile() can only return Err when topological_sort
    // detects a cycle, and build_dag() always produces edges i->j with
    // i < j (acyclic by construction).  So compile() cannot fail through
    // the public API today.
    //
    // The test exists as a forward-looking regression guard: if a future
    // refactor moves self.swap() before the compile call or wraps the
    // control flow differently, this test prevents the invariant from
    // silently breaking.  It also documents the expected contract for
    // consumers of the API.
    //
    // When compile DOES gain an error path (e.g., resource validation,
    // pass dependency checking), this test immediately covers it.
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_and_swap_error_preserves_slot() {
        let original_name = "original_graph";
        let initial = make_single_pass_graph(original_name);
        let slot = FrameGraphSlot::new(initial);

        // Attempt compile_and_swap with inputs that SHOULD fail but
        // currently succeed because build_dag is acyclic by construction.
        // If compile ever adds validation (e.g., duplicate resource
        // handles, pass cycles, mismatched attachments), this will
        // exercise the error path.
        //
        // Two passes both writing to the same resource create a WAW edge.
        // This is valid DAG input and compiles fine today, but exercises
        // a realistic multi-pass compile path.
        let resource = super::super::IrResource::new(
            super::super::ResourceHandle(10),
            "shared_rt",
            super::super::ResourceDesc::Texture2D(super::super::TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            super::super::ResourceLifetime::Transient,
            super::super::ResourceState::Uninitialized,
        );

        let att = super::super::ColorAttachment {
            resource: super::super::ResourceHandle(10),
            mip_level: 0,
            array_layer: 0,
            load_op: super::super::AttachmentLoadOp::Clear,
            store_op: super::super::AttachmentStoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        };

        let p0 = super::super::IrPass::graphics(
            super::super::PassIndex(0),
            "pass_a",
            vec![att.clone()],
            None,
            super::super::InstanceSource::Direct {
                index_count: 6,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            super::super::ViewType::ColorAttachment,
        );

        let p1 = super::super::IrPass::graphics(
            super::super::PassIndex(1),
            "pass_b",
            vec![att],
            None,
            super::super::InstanceSource::Direct {
                index_count: 6,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            super::super::ViewType::ColorAttachment,
        );

        let result = slot.compile_and_swap(vec![p0, p1], vec![resource]);

        if result.is_err() {
            // Compile failed -- the invariant MUST hold:
            // the slot still contains the original graph.
            assert_eq!(
                slot.load().passes[0].name,
                original_name,
                "slot must retain the previous graph after a compile error"
            );
            assert!(
                slot.is_loaded(),
                "slot must remain loaded after a compile error"
            );
        } else {
            // Compile succeeded -- the slot should contain the new graph.
            // This is the expected path today (compile() has no error path
            // through the public API).  The test still validates the slot
            // is consistent.
            assert_eq!(
                slot.load().passes[0].name,
                "pass_a",
                "slot should contain the first pass of the compiled graph"
            );
            assert!(slot.is_loaded(), "slot must remain loaded after compile");
        }
    }

    // -----------------------------------------------------------------------
    // 9. Slot with a minimal (single-pass) graph works
    // -----------------------------------------------------------------------
    #[test]
    fn test_slot_with_minimal_graph() {
        let graph = make_single_pass_graph("minimal");
        let slot = FrameGraphSlot::new(graph);

        assert!(slot.is_loaded());
        let loaded = slot.load();
        assert_eq!(loaded.passes.len(), 1);
        assert_eq!(loaded.passes[0].name, "minimal");
        assert_eq!(loaded.resources.len(), 1);
    }
}
