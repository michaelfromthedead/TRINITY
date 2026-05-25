// SPDX-License-Identifier: MIT
//
// scheduler.rs — Scheduler Bridge: frame-loop dispatching system phases via
// thread pool (T-CORE-5.5)

use crate::checksum::HierarchicalChecksum;
use crate::command_buffer::CommandBuffer;
use crate::component_store::ComponentStore;
use crate::job_graph::JobGraphBuilder;
use crate::system_phase::{PhaseGraph, System, SystemContext, SystemPhase};
use crate::thread_pool::ThreadPool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// FrameContext
// ---------------------------------------------------------------------------

/// Per-frame context produced by [`Scheduler::step`].
#[derive(Debug, Clone)]
pub struct FrameContext {
    /// Time delta in seconds since the last frame.
    pub delta_time: f32,
    /// Monotonically increasing frame counter.
    pub frame_number: u64,
    /// World checksum captured before frame execution.
    pub checksum_before: u64,
    /// World checksum captured after all phases completed.
    pub checksum_after: u64,
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Frame-loop scheduler that owns a [`ThreadPool`], [`PhaseGraph`],
/// [`CommandBuffer`], and [`HierarchicalChecksum`].
///
/// Call [`add_phase`](Self::add_phase) and
/// [`add_dependency`](Self::add_dependency) to set up the pipeline, then
/// call [`step`](Self::step) each frame to advance the loop.
///
/// # Modes
///
/// | Constructor              | Behaviour                                 |
/// |--------------------------|-------------------------------------------|
/// | [`new(n)`]              | Thread pool with `n` workers              |
/// | [`new_auto`]             | Pool auto-sized to available CPUs         |
/// | [`new_single_threaded`] | Sequential execution, no thread pool      |
pub struct Scheduler {
    /// `Some(pool)` for multi-threaded mode, `None` for single-threaded
    /// debug mode.
    pool: Option<ThreadPool>,
    /// Phase graph defining the pipeline topology.
    phase_graph: PhaseGraph,
    /// Deferred ECS command buffer.
    command_buffer: CommandBuffer,
    /// Hierarchical checksum for deterministic state verification.
    checksum: HierarchicalChecksum,
    /// Monotonically increasing frame counter.
    frame_number: u64,
}

impl Scheduler {
    // ── Construction ────────────────────────────────────────────────

    /// Creates a new scheduler with a thread pool of `num_workers` threads.
    ///
    /// When `num_workers` is `0`, the pool is auto-sized to the number of
    /// available CPUs.
    pub fn new(num_workers: usize) -> Self {
        let pool = if num_workers == 0 {
            Some(ThreadPool::new_auto())
        } else {
            Some(ThreadPool::new(num_workers))
        };
        Self::new_with_pool(pool)
    }

    /// Creates a scheduler whose pool auto-sizes to the number of
    /// available CPUs.
    pub fn new_auto() -> Self {
        Self::new(0)
    }

    /// Creates a scheduler in single-threaded debug mode.
    ///
    /// In this mode all phases execute sequentially on the caller's thread
    /// and no thread pool is created.  Useful for debugging, determinism
    /// verification, and platforms that do not support threading.
    pub fn new_single_threaded() -> Self {
        Self::new_with_pool(None)
    }

    /// Shared constructor that all `new_*` methods delegate to.
    fn new_with_pool(pool: Option<ThreadPool>) -> Self {
        Self {
            pool,
            phase_graph: PhaseGraph::new(),
            command_buffer: CommandBuffer::new(),
            checksum: HierarchicalChecksum::new(),
            frame_number: 0,
        }
    }

    // ── Accessors ───────────────────────────────────────────────────

    /// Returns `true` when the scheduler is in single-threaded debug mode.
    pub fn is_single_threaded(&self) -> bool {
        self.pool.is_none()
    }

    /// Shared access to the command buffer.
    pub fn command_buffer(&self) -> &CommandBuffer {
        &self.command_buffer
    }

    /// Mutable access to the command buffer.
    pub fn command_buffer_mut(&mut self) -> &mut CommandBuffer {
        &mut self.command_buffer
    }

    /// Shared access to the hierarchical checksum.
    pub fn checksum(&self) -> &HierarchicalChecksum {
        &self.checksum
    }

    /// Returns the current frame number.
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    // ── Phase registration ──────────────────────────────────────────

    /// Registers a new phase with the given name and systems.
    ///
    /// All systems must be `Send + 'static` to support thread-pool
    /// dispatch.  Systems within a phase always run sequentially in
    /// declaration order; parallelism is achieved across phases.
    pub fn add_phase<S, I>(&mut self, name: &str, systems: I)
    where
        S: System + Send + 'static,
        I: IntoIterator<Item = S>,
    {
        let mut phase = SystemPhase::new(name);
        for system in systems {
            phase.add_system(system);
        }
        self.phase_graph.add_phase(phase);
    }

    /// Declares that `phase` depends on `depends_on`.
    ///
    /// `depends_on` will always run before `phase`.
    ///
    /// # Panics
    ///
    /// Panics if either name has not been registered via
    /// [`add_phase`](Self::add_phase).
    pub fn add_dependency(&mut self, phase: &str, depends_on: &str) {
        self.phase_graph.add_dependency(phase, depends_on);
    }

    // ── Frame stepping ──────────────────────────────────────────────

    /// Advance the frame loop by `delta_time` seconds.
    ///
    /// The per-frame pipeline is:
    ///
    /// 1. Capture the pre-frame checksum.
    /// 2. Reset the hierarchical checksum for the new frame.
    /// 3. Execute all enabled phases in topological order (sequentially
    ///    in single-threaded mode; via [`JobGraph`] + thread pool in
    ///    multi-threaded mode).
    /// 4. Flush the [`CommandBuffer`] to `store`.
    /// 5. Capture the post-frame checksum.
    pub fn step(&mut self, delta_time: f32, store: &mut ComponentStore) -> FrameContext {
        let checksum_before = self.checksum.world_checksum();
        self.checksum.reset();

        let ctx = SystemContext::new(delta_time, self.frame_number);
        self.frame_number += 1;

        // Take the pool out of self so we can borrow self mutably for the
        // phase graph while the pool is a local variable.
        let mut pool_opt = self.pool.take();

        if let Some(ref mut pool) = pool_opt {
            // Multi-threaded mode: build a JobGraph and execute.
            let order = match self.phase_graph.topological_order() {
                Ok(o) => o,
                Err(e) => panic!("Scheduler: phase-graph cycle detected: {e}"),
            };

            if !order.is_empty() {
                // Extract phases from the graph and wrap in Arc<Mutex<>>.
                let mut shared: HashMap<String, Arc<Mutex<SystemPhase>>> = HashMap::new();
                for name in &order {
                    if let Some(phase) = self.phase_graph.remove_phase(name) {
                        shared.insert(name.clone(), Arc::new(Mutex::new(phase)));
                    }
                }

                // Snapshot edges.
                let edges: Vec<(String, String)> =
                    self.phase_graph.edges().cloned().collect();

                // Build JobGraph.
                let mut builder = JobGraphBuilder::new();
                for name in &order {
                    let Some(arc) = shared.get(name) else {
                        continue;
                    };
                    let phase_arc = Arc::clone(arc);
                    let ctx_clone = ctx.clone();
                    builder.add_task(name, move || {
                        phase_arc
                            .lock()
                            .expect("phase mutex is not poisoned")
                            .run_all(&ctx_clone);
                    });
                }

                // Wire up dependency edges.
                // PhaseGraph edges are (phase, depends_on); JobGraph edges
                // are (from, to) — i.e. from must run before to.
                for (phase, dep) in &edges {
                    if shared.contains_key(phase) && shared.contains_key(dep) {
                        builder.depends_on(dep, phase);
                    }
                }

                // Execute — blocks until all tasks complete.
                let graph =
                    builder.finalize().expect("valid JobGraph from phase DAG");
                graph.execute(pool);

                // Restore phases into the graph for subsequent frames.
                for (name, arc) in shared {
                    match Arc::try_unwrap(arc) {
                        Ok(mutex) => {
                            let phase =
                                mutex.into_inner().expect("non-poisoned mutex");
                            self.phase_graph.insert_phase(phase);
                        }
                        Err(_) => {
                            // Should never happen after execute() returns
                            // since all tasks and the JobGraph itself will
                            // have been dropped, releasing every Arc clone.
                            eprintln!(
                                "Scheduler: warning — Arc<Mutex<{name}>> \
                                 still has outstanding references after \
                                 execute"
                            );
                        }
                    }
                }
            }
        } else {
            // Single-threaded debug mode: run phases sequentially on the
            // calling thread via PhaseGraph::execute.
            self.phase_graph.execute(&ctx).unwrap();
        }

        // Restore the pool.
        self.pool = pool_opt;

        // Apply deferred ECS commands.
        self.command_buffer.flush(store);

        let checksum_after = self.checksum.world_checksum();

        FrameContext {
            delta_time,
            frame_number: self.frame_number.wrapping_sub(1),
            checksum_before,
            checksum_after,
        }
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("single_threaded", &self.pool.is_none())
            .field("frame_number", &self.frame_number)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component_store::ComponentStore;
    use crate::system_phase::System;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A system that records its name in a shared log.
    struct Recorder {
        name: &'static str,
        log: Arc<Mutex<Vec<&'static str>>>,
    }

    impl System for Recorder {
        fn run(&mut self, _ctx: &SystemContext) {
            self.log.lock().unwrap().push(self.name);
        }
    }

    /// A system that advances a counter.
    struct Counter {
        ctr: Arc<AtomicUsize>,
    }

    impl System for Counter {
        fn run(&mut self, _ctx: &SystemContext) {
            self.ctr.fetch_add(1, Ordering::SeqCst);
        }
    }

    // ── Single-threaded mode ────────────────────────────────────────

    #[test]
    fn single_threaded_executes_in_order() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase(
            "a",
            vec![Recorder {
                name: "a",
                log: Arc::clone(&log),
            }],
        );
        sched.add_phase(
            "b",
            vec![Recorder {
                name: "b",
                log: Arc::clone(&log),
            }],
        );
        sched.add_dependency("b", "a");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(1.0 / 60.0, &mut store);

        assert_eq!(ctx.delta_time, 1.0 / 60.0);
        assert_eq!(ctx.frame_number, 0);
        assert_eq!(ctx.checksum_before, 0);

        let entries = log.lock().unwrap().clone();
        assert_eq!(entries, vec!["a", "b"]);
    }

    #[test]
    fn single_threaded_frame_number_increments() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("noop", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let c0 = sched.step(0.016, &mut store);
        let c1 = sched.step(0.016, &mut store);
        let c2 = sched.step(0.016, &mut store);

        assert_eq!(c0.frame_number, 0);
        assert_eq!(c1.frame_number, 1);
        assert_eq!(c2.frame_number, 2);
    }

    // ── Multi-threaded mode ─────────────────────────────────────────

    #[test]
    fn parallel_executes_all_phases() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new(2);
        sched.add_phase(
            "a",
            vec![Recorder {
                name: "a",
                log: Arc::clone(&log),
            }],
        );
        sched.add_phase(
            "b",
            vec![Recorder {
                name: "b",
                log: Arc::clone(&log),
            }],
        );
        sched.add_dependency("b", "a");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(1.0 / 60.0, &mut store);

        assert_eq!(ctx.frame_number, 0);
        assert_eq!(ctx.checksum_before, 0);

        let entries = log.lock().unwrap().clone();
        assert!(entries.contains(&"a"));
        assert!(entries.contains(&"b"));
        // "a" must appear before "b" in the log (dependency ordering).
        let pos_a = entries.iter().position(|x| *x == "a").unwrap();
        let pos_b = entries.iter().position(|x| *x == "b").unwrap();
        assert!(pos_a < pos_b, "dependency ordering violated");
    }

    #[test]
    fn parallel_respects_chain_dependency() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new(2);
        for i in 0..5 {
            let l = Arc::clone(&log);
            sched.add_phase(
                &format!("p{i}"),
                vec![Recorder {
                    name: Box::leak(format!("p{i}").into_boxed_str()),
                    log: l,
                }],
            );
            if i > 0 {
                sched.add_dependency(&format!("p{i}"), &format!("p{}", i - 1));
            }
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        assert_eq!(entries.len(), 5);
        // Linear chain: p0 -> p1 -> p2 -> p3 -> p4
        for i in 0..4 {
            let pos_cur = entries.iter().position(|x| *x == format!("p{i}")).unwrap();
            let pos_nxt = entries.iter().position(|x| *x == format!("p{}", i + 1)).unwrap();
            assert!(pos_cur < pos_nxt, "chain ordering violated at p{i}");
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn empty_graph_does_not_panic() {
        let mut sched = Scheduler::new_single_threaded();
        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(1.0, &mut store);
        assert_eq!(ctx.frame_number, 0);

        // Also test with a thread pool.
        let mut sched = Scheduler::new(1);
        let ctx = sched.step(0.5, &mut store);
        assert_eq!(ctx.frame_number, 0);
    }

    #[test]
    fn single_threaded_mode_flag() {
        let s = Scheduler::new_single_threaded();
        assert!(s.is_single_threaded());

        let p = Scheduler::new(2);
        assert!(!p.is_single_threaded());

        let a = Scheduler::new_auto();
        assert!(!a.is_single_threaded());
    }

    #[test]
    fn multiple_frames_accumulate_counter() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(2);
        sched.add_phase(
            "count",
            vec![Counter {
                ctr: Arc::clone(&ctr),
            }],
        );

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        for _ in 0..10 {
            sched.step(0.016, &mut store);
        }

        assert_eq!(ctr.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn checksum_before_is_previous_frames_after() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("noop", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let c0 = sched.step(0.016, &mut store);
        let c1 = sched.step(0.016, &mut store);

        // The "before" of frame 1 should equal the "after" of frame 0
        // when no checksum updates happened during the frame.
        assert_eq!(c1.checksum_before, c0.checksum_after);
    }
}
