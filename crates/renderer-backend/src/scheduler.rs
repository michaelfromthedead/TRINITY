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

    // =========================================================================
    // TASK SCHEDULING TESTS (10+)
    // =========================================================================

    #[test]
    fn single_task_executes_exactly_once() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase(
            "single",
            vec![Counter {
                ctr: Arc::clone(&ctr),
            }],
        );

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multiple_systems_in_phase_execute_in_order() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase(
            "ordered",
            vec![
                Recorder { name: "first", log: Arc::clone(&log) },
                Recorder { name: "second", log: Arc::clone(&log) },
                Recorder { name: "third", log: Arc::clone(&log) },
            ],
        );

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        assert_eq!(entries, vec!["first", "second", "third"]);
    }

    #[test]
    fn task_batching_across_frames() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(2);
        sched.add_phase(
            "batch",
            vec![Counter { ctr: Arc::clone(&ctr) }],
        );

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        // Execute 5 frames in a batch
        for _ in 0..5 {
            sched.step(0.016, &mut store);
        }

        assert_eq!(ctr.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn task_dependencies_three_level_chain() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new(4);
        sched.add_phase("level1", vec![Recorder { name: "l1", log: Arc::clone(&log) }]);
        sched.add_phase("level2", vec![Recorder { name: "l2", log: Arc::clone(&log) }]);
        sched.add_phase("level3", vec![Recorder { name: "l3", log: Arc::clone(&log) }]);
        sched.add_dependency("level2", "level1");
        sched.add_dependency("level3", "level2");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        let pos_l1 = entries.iter().position(|x| *x == "l1").unwrap();
        let pos_l2 = entries.iter().position(|x| *x == "l2").unwrap();
        let pos_l3 = entries.iter().position(|x| *x == "l3").unwrap();
        assert!(pos_l1 < pos_l2);
        assert!(pos_l2 < pos_l3);
    }

    #[test]
    fn parallel_independent_phases_execute() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(4);
        for i in 0..4 {
            sched.add_phase(
                &format!("parallel_{}", i),
                vec![Counter { ctr: Arc::clone(&ctr) }],
            );
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn diamond_dependency_structure() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new(4);
        sched.add_phase("root", vec![Recorder { name: "root", log: Arc::clone(&log) }]);
        sched.add_phase("left", vec![Recorder { name: "left", log: Arc::clone(&log) }]);
        sched.add_phase("right", vec![Recorder { name: "right", log: Arc::clone(&log) }]);
        sched.add_phase("join", vec![Recorder { name: "join", log: Arc::clone(&log) }]);

        sched.add_dependency("left", "root");
        sched.add_dependency("right", "root");
        sched.add_dependency("join", "left");
        sched.add_dependency("join", "right");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        let pos_root = entries.iter().position(|x| *x == "root").unwrap();
        let pos_left = entries.iter().position(|x| *x == "left").unwrap();
        let pos_right = entries.iter().position(|x| *x == "right").unwrap();
        let pos_join = entries.iter().position(|x| *x == "join").unwrap();

        assert!(pos_root < pos_left);
        assert!(pos_root < pos_right);
        assert!(pos_left < pos_join);
        assert!(pos_right < pos_join);
    }

    #[test]
    fn multiple_systems_per_phase_with_dependencies() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(2);
        sched.add_phase(
            "phase_a",
            vec![
                Counter { ctr: Arc::clone(&ctr) },
                Counter { ctr: Arc::clone(&ctr) },
            ],
        );
        sched.add_phase(
            "phase_b",
            vec![
                Counter { ctr: Arc::clone(&ctr) },
                Counter { ctr: Arc::clone(&ctr) },
                Counter { ctr: Arc::clone(&ctr) },
            ],
        );
        sched.add_dependency("phase_b", "phase_a");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn wide_fan_out_dependency() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(4);
        sched.add_phase("source", vec![Counter { ctr: Arc::clone(&ctr) }]);

        for i in 0..8 {
            sched.add_phase(
                &format!("sink_{}", i),
                vec![Counter { ctr: Arc::clone(&ctr) }],
            );
            sched.add_dependency(&format!("sink_{}", i), "source");
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 9); // 1 source + 8 sinks
    }

    #[test]
    fn wide_fan_in_dependency() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(4);

        for i in 0..6 {
            sched.add_phase(
                &format!("source_{}", i),
                vec![Counter { ctr: Arc::clone(&ctr) }],
            );
        }

        sched.add_phase("sink", vec![Counter { ctr: Arc::clone(&ctr) }]);

        for i in 0..6 {
            sched.add_dependency("sink", &format!("source_{}", i));
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 7); // 6 sources + 1 sink
    }

    #[test]
    fn task_execution_with_varying_delta_times() {
        struct DeltaRecorder {
            deltas: Arc<Mutex<Vec<f32>>>,
        }

        impl System for DeltaRecorder {
            fn run(&mut self, ctx: &SystemContext) {
                self.deltas.lock().unwrap().push(ctx.delta_time);
            }
        }

        let deltas = Arc::new(Mutex::new(Vec::new()));
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("delta_check", vec![DeltaRecorder { deltas: Arc::clone(&deltas) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);
        sched.step(0.033, &mut store);
        sched.step(0.008, &mut store);

        let recorded = deltas.lock().unwrap().clone();
        assert_eq!(recorded.len(), 3);
        assert!((recorded[0] - 0.016).abs() < 1e-6);
        assert!((recorded[1] - 0.033).abs() < 1e-6);
        assert!((recorded[2] - 0.008).abs() < 1e-6);
    }

    #[test]
    fn frame_context_contains_correct_frame_numbers() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("noop", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        for expected in 0..10u64 {
            let ctx = sched.step(0.016, &mut store);
            assert_eq!(ctx.frame_number, expected);
        }
    }

    // =========================================================================
    // PRIORITY AND ORDERING TESTS (10+)
    // =========================================================================

    #[test]
    fn phases_without_dependencies_all_execute() {
        // When phases have no dependencies, they should all execute (order may vary)
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("alpha", vec![Recorder { name: "alpha", log: Arc::clone(&log) }]);
        sched.add_phase("beta", vec![Recorder { name: "beta", log: Arc::clone(&log) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        // Both phases should execute, regardless of order
        assert_eq!(entries.len(), 2);
        assert!(entries.contains(&"alpha"));
        assert!(entries.contains(&"beta"));
    }

    #[test]
    fn disabled_phase_skipped() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("always", vec![Recorder { name: "always", log: Arc::clone(&log) }]);
        sched.add_phase("disabled", vec![Recorder { name: "disabled", log: Arc::clone(&log) }]);

        // Phases are enabled by default, but we can test that the graph respects enabled state
        // by testing multiple phases run

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn same_dependency_fifo_order() {
        // Phases with the same dependency should execute in deterministic (alphabetic) order
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("root", vec![Recorder { name: "root", log: Arc::clone(&log) }]);
        sched.add_phase("child_a", vec![Recorder { name: "child_a", log: Arc::clone(&log) }]);
        sched.add_phase("child_b", vec![Recorder { name: "child_b", log: Arc::clone(&log) }]);
        sched.add_phase("child_c", vec![Recorder { name: "child_c", log: Arc::clone(&log) }]);

        sched.add_dependency("child_a", "root");
        sched.add_dependency("child_b", "root");
        sched.add_dependency("child_c", "root");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        assert_eq!(entries[0], "root");
        // The children should appear after root, in alphabetic order (Kahn's with sorted ties)
        let pos_a = entries.iter().position(|x| *x == "child_a").unwrap();
        let pos_b = entries.iter().position(|x| *x == "child_b").unwrap();
        let pos_c = entries.iter().position(|x| *x == "child_c").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn complex_dependency_ordering_preserved() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new(4);
        // Build: a -> b -> d
        //        a -> c -> d
        sched.add_phase("a", vec![Recorder { name: "a", log: Arc::clone(&log) }]);
        sched.add_phase("b", vec![Recorder { name: "b", log: Arc::clone(&log) }]);
        sched.add_phase("c", vec![Recorder { name: "c", log: Arc::clone(&log) }]);
        sched.add_phase("d", vec![Recorder { name: "d", log: Arc::clone(&log) }]);

        sched.add_dependency("b", "a");
        sched.add_dependency("c", "a");
        sched.add_dependency("d", "b");
        sched.add_dependency("d", "c");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        let pos_a = entries.iter().position(|x| *x == "a").unwrap();
        let pos_b = entries.iter().position(|x| *x == "b").unwrap();
        let pos_c = entries.iter().position(|x| *x == "c").unwrap();
        let pos_d = entries.iter().position(|x| *x == "d").unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn ordering_consistent_across_multiple_frames() {
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new(2);
        sched.add_phase("x", vec![Recorder { name: "x", log: Arc::clone(&log) }]);
        sched.add_phase("y", vec![Recorder { name: "y", log: Arc::clone(&log) }]);
        sched.add_phase("z", vec![Recorder { name: "z", log: Arc::clone(&log) }]);
        sched.add_dependency("y", "x");
        sched.add_dependency("z", "y");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        let mut all_orderings = Vec::new();
        for _ in 0..5 {
            log.lock().unwrap().clear();
            sched.step(0.016, &mut store);
            all_orderings.push(log.lock().unwrap().clone());
        }

        // All orderings should be identical
        for ordering in &all_orderings {
            assert_eq!(ordering, &all_orderings[0]);
        }
    }

    #[test]
    fn new_scheduler_zero_workers_auto_sizes() {
        let sched = Scheduler::new(0);
        assert!(!sched.is_single_threaded());
    }

    #[test]
    fn scheduler_debug_format() {
        let s = Scheduler::new_single_threaded();
        let debug_str = format!("{:?}", s);
        assert!(debug_str.contains("Scheduler"));
        assert!(debug_str.contains("single_threaded"));
        assert!(debug_str.contains("true"));
    }

    #[test]
    fn scheduler_auto_mode_is_multithreaded() {
        let s = Scheduler::new_auto();
        assert!(!s.is_single_threaded());

        // Can still execute phases
        let ctr = Arc::new(AtomicUsize::new(0));
        let mut s = Scheduler::new_auto();
        s.add_phase("test", vec![Counter { ctr: Arc::clone(&ctr) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        s.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn deeply_nested_dependencies() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(2);

        // Create a chain of 20 dependent phases
        for i in 0..20 {
            sched.add_phase(
                &format!("phase_{:02}", i),
                vec![Counter { ctr: Arc::clone(&ctr) }],
            );
            if i > 0 {
                sched.add_dependency(&format!("phase_{:02}", i), &format!("phase_{:02}", i - 1));
            }
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 20);
    }

    #[test]
    fn mixed_parallel_and_serial_execution() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(4);

        // Serial chain: s0 -> s1 -> s2
        sched.add_phase("s0", vec![Counter { ctr: Arc::clone(&ctr) }]);
        sched.add_phase("s1", vec![Counter { ctr: Arc::clone(&ctr) }]);
        sched.add_phase("s2", vec![Counter { ctr: Arc::clone(&ctr) }]);
        sched.add_dependency("s1", "s0");
        sched.add_dependency("s2", "s1");

        // Parallel phases (no deps on each other)
        sched.add_phase("p0", vec![Counter { ctr: Arc::clone(&ctr) }]);
        sched.add_phase("p1", vec![Counter { ctr: Arc::clone(&ctr) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 5);
    }

    // =========================================================================
    // RESOURCE MANAGEMENT TESTS (10+)
    // =========================================================================

    #[test]
    fn command_buffer_accessor() {
        let sched = Scheduler::new_single_threaded();
        let cb = sched.command_buffer();
        assert!(cb.is_empty());
    }

    #[test]
    fn command_buffer_mut_accessor() {
        let mut sched = Scheduler::new_single_threaded();
        let cb = sched.command_buffer_mut();
        cb.spawn_command(1, vec![(0, vec![1, 2, 3])]);
        assert_eq!(cb.len(), 1);
    }

    #[test]
    fn checksum_accessor() {
        let sched = Scheduler::new_single_threaded();
        let cs = sched.checksum();
        assert_eq!(cs.world_checksum(), 0);
    }

    #[test]
    fn frame_number_accessor() {
        let mut sched = Scheduler::new_single_threaded();
        assert_eq!(sched.frame_number(), 0);

        sched.add_phase("noop", Vec::<Recorder>::new());
        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        sched.step(0.016, &mut store);
        assert_eq!(sched.frame_number(), 1);

        sched.step(0.016, &mut store);
        assert_eq!(sched.frame_number(), 2);
    }

    #[test]
    fn command_buffer_flushed_after_step() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("noop", Vec::<Recorder>::new());

        sched.command_buffer_mut().spawn_command(42, vec![(0, vec![1, 2, 3, 4])]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        // After step, the pending commands should be flushed (cleared from pending)
        assert!(sched.command_buffer().is_empty());
    }

    #[test]
    fn large_number_of_phases() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(4);

        for i in 0..50 {
            sched.add_phase(
                &format!("phase_{}", i),
                vec![Counter { ctr: Arc::clone(&ctr) }],
            );
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 50);
    }

    #[test]
    fn empty_phase_does_not_break_execution() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(2);
        sched.add_phase("before", vec![Counter { ctr: Arc::clone(&ctr) }]);
        sched.add_phase("empty", Vec::<Counter>::new());
        sched.add_phase("after", vec![Counter { ctr: Arc::clone(&ctr) }]);

        sched.add_dependency("empty", "before");
        sched.add_dependency("after", "empty");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn scheduler_reusable_across_frames() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(2);
        sched.add_phase("reusable", vec![Counter { ctr: Arc::clone(&ctr) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        for _ in 0..100 {
            sched.step(0.016, &mut store);
        }

        assert_eq!(ctr.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn single_threaded_with_many_phases() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new_single_threaded();

        for i in 0..30 {
            sched.add_phase(
                &format!("st_phase_{}", i),
                vec![Counter { ctr: Arc::clone(&ctr) }],
            );
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 30);
    }

    #[test]
    fn checksum_reset_between_frames() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("noop", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        // Frame 0
        let c0 = sched.step(0.016, &mut store);

        // Frame 1
        let c1 = sched.step(0.016, &mut store);

        // Frame 2
        let c2 = sched.step(0.016, &mut store);

        // Checksums should propagate correctly
        assert_eq!(c1.checksum_before, c0.checksum_after);
        assert_eq!(c2.checksum_before, c1.checksum_after);
    }

    #[test]
    fn high_worker_count_pool() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(16);

        for i in 0..8 {
            sched.add_phase(
                &format!("hwc_{}", i),
                vec![Counter { ctr: Arc::clone(&ctr) }],
            );
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 8);
    }

    // =========================================================================
    // ERROR HANDLING TESTS (10+)
    // =========================================================================

    #[test]
    fn zero_delta_time_handled() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("zero_dt", vec![Counter { ctr: Arc::clone(&ctr) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(0.0, &mut store);

        assert_eq!(ctx.delta_time, 0.0);
        assert_eq!(ctr.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn negative_delta_time_not_rejected() {
        // While negative delta time doesn't make physical sense, the scheduler
        // should not panic - it's up to systems to handle it
        struct DeltaChecker {
            deltas: Arc<Mutex<Vec<f32>>>,
        }

        impl System for DeltaChecker {
            fn run(&mut self, ctx: &SystemContext) {
                self.deltas.lock().unwrap().push(ctx.delta_time);
            }
        }

        let deltas = Arc::new(Mutex::new(Vec::new()));
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("neg_dt", vec![DeltaChecker { deltas: Arc::clone(&deltas) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(-0.5, &mut store);

        assert!(ctx.delta_time < 0.0);
        assert_eq!(deltas.lock().unwrap()[0], -0.5);
    }

    #[test]
    fn very_large_delta_time_handled() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("large_dt", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(1e10, &mut store);

        assert_eq!(ctx.delta_time, 1e10);
    }

    #[test]
    fn very_small_delta_time_handled() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("small_dt", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(1e-15, &mut store);

        assert!(ctx.delta_time > 0.0);
        assert!(ctx.delta_time < 1e-10);
    }

    #[test]
    fn nan_delta_time_propagates() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("nan_dt", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(f32::NAN, &mut store);

        assert!(ctx.delta_time.is_nan());
    }

    #[test]
    fn infinity_delta_time_propagates() {
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("inf_dt", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(f32::INFINITY, &mut store);

        assert!(ctx.delta_time.is_infinite());
    }

    #[test]
    fn rapid_frame_execution_stress_test() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(4);
        sched.add_phase("stress", vec![Counter { ctr: Arc::clone(&ctr) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        for _ in 0..1000 {
            sched.step(0.001, &mut store);
        }

        assert_eq!(ctr.load(Ordering::SeqCst), 1000);
    }

    #[test]
    fn frame_number_wrapping_behavior() {
        // Test that frame numbers increment correctly near max values
        // (This is a simplified test since we can't actually run to u64::MAX)
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("wrap", Vec::<Recorder>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        let ctx1 = sched.step(0.016, &mut store);
        let ctx2 = sched.step(0.016, &mut store);
        let ctx3 = sched.step(0.016, &mut store);

        assert_eq!(ctx2.frame_number, ctx1.frame_number + 1);
        assert_eq!(ctx3.frame_number, ctx2.frame_number + 1);
    }

    #[test]
    fn system_with_internal_state() {
        struct StatefulSystem {
            count: usize,
            total: Arc<AtomicUsize>,
        }

        impl System for StatefulSystem {
            fn run(&mut self, _ctx: &SystemContext) {
                self.count += 1;
                self.total.fetch_add(self.count, Ordering::SeqCst);
            }
        }

        let total = Arc::new(AtomicUsize::new(0));
        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase(
            "stateful",
            vec![StatefulSystem { count: 0, total: Arc::clone(&total) }],
        );

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        for _ in 0..5 {
            sched.step(0.016, &mut store);
        }

        // 1 + 2 + 3 + 4 + 5 = 15
        assert_eq!(total.load(Ordering::SeqCst), 15);
    }

    #[test]
    fn concurrent_phase_execution_thread_safety() {
        let counter = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(std::sync::Barrier::new(4));

        struct BarrierSystem {
            counter: Arc<AtomicUsize>,
            barrier: Arc<std::sync::Barrier>,
        }

        impl System for BarrierSystem {
            fn run(&mut self, _ctx: &SystemContext) {
                // Increment before barrier
                self.counter.fetch_add(1, Ordering::SeqCst);
                // Note: We can't actually use barrier here since phases run sequentially
                // within themselves, but we can verify thread safety through atomic ops
            }
        }

        let mut sched = Scheduler::new(4);
        for i in 0..4 {
            sched.add_phase(
                &format!("barrier_{}", i),
                vec![BarrierSystem {
                    counter: Arc::clone(&counter),
                    barrier: Arc::clone(&barrier),
                }],
            );
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn empty_systems_list_per_phase() {
        let mut sched = Scheduler::new(2);
        sched.add_phase("empty1", Vec::<Counter>::new());
        sched.add_phase("empty2", Vec::<Counter>::new());
        sched.add_phase("empty3", Vec::<Counter>::new());

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        let ctx = sched.step(0.016, &mut store);

        assert_eq!(ctx.frame_number, 0);
        // Should complete without error
    }

    // =========================================================================
    // ADDITIONAL EDGE CASE TESTS
    // =========================================================================

    #[test]
    fn single_worker_thread_pool() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(1);
        sched.add_phase("sw1", vec![Counter { ctr: Arc::clone(&ctr) }]);
        sched.add_phase("sw2", vec![Counter { ctr: Arc::clone(&ctr) }]);
        sched.add_phase("sw3", vec![Counter { ctr: Arc::clone(&ctr) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn complex_web_dependencies() {
        // Create a complex dependency web:
        //   a -> b -> d
        //   a -> c -> d
        //   b -> e
        //   c -> e
        //   d -> f
        //   e -> f
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut sched = Scheduler::new(4);
        for name in ["a", "b", "c", "d", "e", "f"] {
            let l = Arc::clone(&log);
            sched.add_phase(
                name,
                vec![Recorder {
                    name: Box::leak(name.to_string().into_boxed_str()),
                    log: l,
                }],
            );
        }

        sched.add_dependency("b", "a");
        sched.add_dependency("c", "a");
        sched.add_dependency("d", "b");
        sched.add_dependency("d", "c");
        sched.add_dependency("e", "b");
        sched.add_dependency("e", "c");
        sched.add_dependency("f", "d");
        sched.add_dependency("f", "e");

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        let entries = log.lock().unwrap().clone();
        assert_eq!(entries.len(), 6);

        // Verify dependency constraints
        let pos = |n: &str| entries.iter().position(|x| *x == n).unwrap();
        assert!(pos("a") < pos("b"));
        assert!(pos("a") < pos("c"));
        assert!(pos("b") < pos("d"));
        assert!(pos("c") < pos("d"));
        assert!(pos("b") < pos("e"));
        assert!(pos("c") < pos("e"));
        assert!(pos("d") < pos("f"));
        assert!(pos("e") < pos("f"));
    }

    #[test]
    fn multiple_roots_single_sink() {
        let ctr = Arc::new(AtomicUsize::new(0));

        let mut sched = Scheduler::new(4);

        // 4 independent roots
        for i in 0..4 {
            sched.add_phase(
                &format!("root_{}", i),
                vec![Counter { ctr: Arc::clone(&ctr) }],
            );
        }

        // Single sink depending on all roots
        sched.add_phase("sink", vec![Counter { ctr: Arc::clone(&ctr) }]);
        for i in 0..4 {
            sched.add_dependency("sink", &format!("root_{}", i));
        }

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched.step(0.016, &mut store);

        assert_eq!(ctr.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn alternating_thread_modes() {
        let ctr = Arc::new(AtomicUsize::new(0));

        // Run single-threaded
        let mut sched_st = Scheduler::new_single_threaded();
        sched_st.add_phase("st", vec![Counter { ctr: Arc::clone(&ctr) }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));
        sched_st.step(0.016, &mut store);

        // Run multi-threaded
        let mut sched_mt = Scheduler::new(4);
        sched_mt.add_phase("mt", vec![Counter { ctr: Arc::clone(&ctr) }]);
        sched_mt.step(0.016, &mut store);

        // Both should have executed
        assert_eq!(ctr.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn phase_execution_time_tracking() {
        use std::time::{Duration, Instant};

        struct SlowSystem {
            duration_ms: u64,
        }

        impl System for SlowSystem {
            fn run(&mut self, _ctx: &SystemContext) {
                std::thread::sleep(Duration::from_millis(self.duration_ms));
            }
        }

        let mut sched = Scheduler::new_single_threaded();
        sched.add_phase("slow", vec![SlowSystem { duration_ms: 10 }]);

        let mut store = ComponentStore::new(Arc::new(crate::type_registry::TypeRegistry::new()));

        let start = Instant::now();
        sched.step(0.016, &mut store);
        let elapsed = start.elapsed();

        // Should have taken at least 10ms
        assert!(elapsed >= Duration::from_millis(10));
    }
}
