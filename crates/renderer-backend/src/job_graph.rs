// SPDX-License-Identifier: MIT
//
// job_graph.rs — Compiled dependency DAG for task execution (T-CORE-3.2)

use crate::thread_pool::{Priority, ThreadPool};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by [`JobGraphBuilder::finalize`] when the dependency graph
/// contains a cycle or references unknown tasks.
#[derive(Debug, Clone)]
pub struct CycleError {
    /// Task names that form the cycle (or the first detected problem).
    pub cycle: Vec<String>,
}

impl std::fmt::Display for CycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dependency cycle detected: {}", self.cycle.join(" -> "))
    }
}

impl std::error::Error for CycleError {}

// ---------------------------------------------------------------------------
// TaskHandle
// ---------------------------------------------------------------------------

/// A non-blocking handle that tracks whether a single task has completed.
pub struct TaskHandle {
    done: Arc<AtomicBool>,
}

impl TaskHandle {
    /// Returns `true` if the task has finished executing.
    pub fn is_complete(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Blocks until the task completes (spin-yield).
    pub fn wait(&self) {
        while !self.is_complete() {
            std::thread::yield_now();
        }
    }
}

// ---------------------------------------------------------------------------
// JobGraphBuilder
// ---------------------------------------------------------------------------

/// Builder that constructs a [`JobGraph`] by registering named tasks and
/// declaring dependency edges between them.
///
/// # Example
///
/// ```ignore
/// let mut builder = JobGraphBuilder::new();
/// builder
///     .add_task("fetch",  || fetch_data())
///     .add_task("process", || process_data())
///     .depends_on("fetch", "process");
/// let graph = builder.finalize().unwrap();
/// ```
pub struct JobGraphBuilder {
    callables: HashMap<String, Box<dyn FnOnce() + Send + 'static>>,
    deps: HashMap<String, Vec<String>>,
    dependents: HashMap<String, Vec<String>>,
}

impl JobGraphBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        JobGraphBuilder {
            callables: HashMap::new(),
            deps: HashMap::new(),
            dependents: HashMap::new(),
        }
    }

    /// Register a named task.
    ///
    /// If a task with the same name already exists it is silently overwritten.
    pub fn add_task<F>(&mut self, name: &str, task: F) -> &mut Self
    where
        F: FnOnce() + Send + 'static,
    {
        self.callables.insert(name.to_string(), Box::new(task));
        self.deps.entry(name.to_string()).or_default();
        self.dependents.entry(name.to_string()).or_default();
        self
    }

    /// Declare that `from` must finish before `to` may start.
    ///
    /// Both task names must eventually be registered via [`add_task`] before
    /// [`finalize`](Self::finalize) is called, or an error is returned.
    pub fn depends_on(&mut self, from: &str, to: &str) -> &mut Self {
        self.deps.entry(to.to_string()).or_default().push(from.to_string());
        self.dependents
            .entry(from.to_string())
            .or_default()
            .push(to.to_string());
        self
    }

    /// Validate the graph and produce a compiled [`JobGraph`].
    ///
    /// Returns `Err(CycleError)` if the dependency graph contains a cycle or
    /// references tasks that were never added.
    pub fn finalize(self) -> Result<JobGraph, CycleError> {
        // -- Validate that every referenced name was added as a task ----------
        let valid: std::collections::HashSet<&str> =
            self.callables.keys().map(|s| s.as_str()).collect();

        for (task, list) in &self.deps {
            for dep in list {
                if !valid.contains(dep.as_str()) {
                    return Err(CycleError {
                        cycle: vec![format!(
                            "unknown dependency '{}' referenced by '{}'",
                            dep, task
                        )],
                    });
                }
            }
        }
        for (task, list) in &self.dependents {
            for dep in list {
                if !valid.contains(dep.as_str()) {
                    return Err(CycleError {
                        cycle: vec![format!(
                            "unknown dependent '{}' referenced by '{}'",
                            dep, task
                        )],
                    });
                }
            }
        }

        // -- Topological sort with cycle detection ----------------------------
        let all_names: Vec<String> = self.callables.keys().cloned().collect();
        let order = topological_sort(&all_names, &self.deps, &self.dependents)?;

        let indices: HashMap<String, usize> = order
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        Ok(JobGraph {
            callables: self.callables,
            deps: self.deps,
            dependents: self.dependents,
            order,
            indices,
        })
    }
}

impl Default for JobGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JobGraph (compiled DAG)
// ---------------------------------------------------------------------------

/// A compiled, ready-to-execute dependency graph of named tasks.
///
/// Create one via [`JobGraphBuilder`], then call [`execute`](JobGraph::execute)
/// to submit the tasks to a [`ThreadPool`] in dependency order.
pub struct JobGraph {
    callables: HashMap<String, Box<dyn FnOnce() + Send + 'static>>,
    deps: HashMap<String, Vec<String>>,
    dependents: HashMap<String, Vec<String>>,
    order: Vec<String>,
    indices: HashMap<String, usize>,
}

impl JobGraph {
    /// Execute every task on `pool`, respecting declared dependencies, and
    /// return a [`TaskHandle`] for each named task.
    ///
    /// This method **blocks** the caller until all tasks have completed.
    /// If any task panics the panic propagates through the worker thread and
    /// may cause the orchestrator to hang.
    pub fn execute(mut self, pool: &ThreadPool) -> HashMap<String, TaskHandle> {
        let n = self.order.len();
        if n == 0 {
            return HashMap::new();
        }

        // --- Extract callables (consumed exactly once) -----------------------
        let mut task_fns: Vec<Option<Box<dyn FnOnce() + Send + 'static>>> =
            Vec::with_capacity(n);
        for name in &self.order {
            task_fns.push(Some(
                self.callables
                    .remove(name)
                    .expect("every task in the order must have a callable"),
            ));
        }

        // --- Completion flags per task ---------------------------------------
        let flags: Vec<Arc<AtomicBool>> =
            (0..n).map(|_| Arc::new(AtomicBool::new(false))).collect();

        // --- Remaining-dependency counters (main-thread only) ----------------
        let mut deps_rem: Vec<usize> = self
            .order
            .iter()
            .map(|name| self.deps[name].len())
            .collect();

        // --- Dependents as index-lists for fast lookup -----------------------
        let dependents: Vec<Vec<usize>> = self
            .order
            .iter()
            .map(|name| {
                self.dependents
                    .get(name)
                    .map(|list| list.iter().map(|dep| self.indices[dep]).collect())
                    .unwrap_or_default()
            })
            .collect();

        // --- Build handles map -----------------------------------------------
        let mut handles = HashMap::with_capacity(n);
        for (i, name) in self.order.iter().enumerate() {
            handles.insert(name.clone(), TaskHandle {
                done: Arc::clone(&flags[i]),
            });
        }

        // --- Completion notification channel ---------------------------------
        let (done_tx, done_rx) = mpsc::channel::<usize>();

        // --- Submit every ready (root) task ----------------------------------
        for i in 0..n {
            if deps_rem[i] == 0 {
                let f = task_fns[i]
                    .take()
                    .expect("root-task callable must be present");
                let tx = done_tx.clone();
                let flag = Arc::clone(&flags[i]);
                pool.spawn(Priority::Normal, move || {
                    f();
                    flag.store(true, Ordering::Release);
                    let _ = tx.send(i);
                });
            }
        }

        // --- Orchestration loop: process completions, submit newly-ready -----
        let mut completed: usize = 0;
        while completed < n {
            let idx = done_rx.recv().expect("task-completion channel");
            completed += 1;

            for &dep_idx in &dependents[idx] {
                deps_rem[dep_idx] = deps_rem[dep_idx].wrapping_sub(1);
                if deps_rem[dep_idx] == 0 {
                    let f = task_fns[dep_idx]
                        .take()
                        .expect("dependent-task callable must be present");
                    let tx = done_tx.clone();
                    let flag = Arc::clone(&flags[dep_idx]);
                    pool.spawn(Priority::Normal, move || {
                        f();
                        flag.store(true, Ordering::Release);
                        let _ = tx.send(dep_idx);
                    });
                }
            }
        }

        handles
    }
}

// ---------------------------------------------------------------------------
// Topological sort (Kahn's algorithm)
// ---------------------------------------------------------------------------

/// Kahn's BFS-based topological sort.
///
/// Returns `Ok(order)` when the graph is acyclic, or `Err(CycleError)` with
/// the names of every node that still has unfulfilled incoming edges.
fn topological_sort(
    names: &[String],
    deps: &HashMap<String, Vec<String>>,
    dependents: &HashMap<String, Vec<String>>,
) -> Result<Vec<String>, CycleError> {
    let mut in_degree: HashMap<&str, usize> = names
        .iter()
        .map(|n| (n.as_str(), deps.get(n).map_or(0, |d| d.len())))
        .collect();

    let mut queue: VecDeque<&str> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&name, _)| name)
        .collect();

    let mut result = Vec::with_capacity(names.len());
    while let Some(name) = queue.pop_front() {
        result.push(name.to_string());
        if let Some(children) = dependents.get(name) {
            for child in children {
                let deg = in_degree
                    .get_mut(child.as_str())
                    .expect("child must be in name set");
                *deg = deg.wrapping_sub(1);
                if *deg == 0 {
                    queue.push_back(child);
                }
            }
        }
    }

    if result.len() == names.len() {
        Ok(result)
    } else {
        let cycle: Vec<String> = in_degree
            .into_iter()
            .filter(|(_, deg)| *deg > 0)
            .map(|(name, _)| name.to_string())
            .collect();
        Err(CycleError { cycle })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    // =========================================================================
    // Original tests (7 tests)
    // =========================================================================

    #[test]
    fn linear_chain() {
        let pool = ThreadPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();
        let c1 = Arc::clone(&counter);
        builder.add_task("a", move || { c1.store(1, Ordering::SeqCst); });
        let c2 = Arc::clone(&counter);
        builder.add_task("b", move || { assert_eq!(c2.load(Ordering::SeqCst), 1); });
        builder.depends_on("a", "b");

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        pool.shutdown();
    }

    #[test]
    fn diamond_dependency() {
        let pool = ThreadPool::new(4);
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut builder = JobGraphBuilder::new();
        let l1 = Arc::clone(&log);
        builder.add_task("a", move || { l1.lock().unwrap().push("a"); });
        let l2 = Arc::clone(&log);
        builder.add_task("b", move || { l2.lock().unwrap().push("b"); });
        let l3 = Arc::clone(&log);
        builder.add_task("c", move || { l3.lock().unwrap().push("c"); });
        let l4 = Arc::clone(&log);
        builder.add_task("d", move || { l4.lock().unwrap().push("d"); });

        builder
            .depends_on("a", "b")
            .depends_on("a", "c")
            .depends_on("b", "d")
            .depends_on("c", "d");

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        let entries = log.lock().unwrap().clone();
        assert!(entries.contains(&"a"));
        assert!(entries.contains(&"b"));
        assert!(entries.contains(&"c"));
        assert!(entries.contains(&"d"));
        // "a" must appear before "b" and "c"
        let pos_a = entries.iter().position(|x| *x == "a").unwrap();
        let pos_b = entries.iter().position(|x| *x == "b").unwrap();
        let pos_c = entries.iter().position(|x| *x == "c").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        // "b" and "c" must appear before "d"
        let pos_d = entries.iter().position(|x| *x == "d").unwrap();
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);

        pool.shutdown();
    }

    #[test]
    fn cycle_detected() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("a", || {});
        builder.add_task("b", || {});
        builder.add_task("c", || {});
        builder.depends_on("a", "b").depends_on("b", "c").depends_on("c", "a");

        match builder.finalize() {
            Err(e) => assert!(!e.cycle.is_empty(), "cycle error must name tasks"),
            Ok(_) => panic!("expected CycleError"),
        }
    }

    #[test]
    fn unknown_dependency_rejected() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("a", || {});
        builder.depends_on("a", "nonexistent");

        match builder.finalize() {
            Err(e) => assert!(e.cycle[0].contains("nonexistent")),
            Ok(_) => panic!("expected error for unknown task"),
        }
    }

    #[test]
    fn task_handle_tracks_completion() {
        let pool = ThreadPool::new(2);

        let mut builder = JobGraphBuilder::new();
        builder.add_task("a", || { std::thread::sleep(std::time::Duration::from_millis(5)); });
        builder.add_task("b", || {});
        builder.depends_on("a", "b");

        let graph = builder.finalize().unwrap();
        let handles = graph.execute(&pool);

        // "a" must eventually be complete
        handles["a"].wait();
        assert!(handles["a"].is_complete());
        assert!(handles["b"].is_complete());

        pool.shutdown();
    }

    #[test]
    fn empty_graph() {
        let pool = ThreadPool::new(2);
        let builder = JobGraphBuilder::new();
        let graph = builder.finalize().unwrap();
        let handles = graph.execute(&pool);
        assert!(handles.is_empty());
        pool.shutdown();
    }

    #[test]
    fn fan_out_fan_in() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();
        builder.add_task("root", || {});

        for i in 0..5 {
            let c = Arc::clone(&counter);
            builder.add_task(&format!("mid_{}", i), move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
            builder.depends_on("root", &format!("mid_{}", i));
        }

        let c = Arc::clone(&counter);
        builder.add_task("join", move || {
            assert_eq!(c.load(Ordering::SeqCst), 5);
        });
        for i in 0..5 {
            builder.depends_on(&format!("mid_{}", i), "join");
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);
        // Counter was incremented 5 times by the mid tasks, then join asserted.
        // If we get here without panic, it worked.
        pool.shutdown();
    }

    // =========================================================================
    // GRAPH CONSTRUCTION TESTS (10+ tests)
    // =========================================================================

    #[test]
    fn add_single_node() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("single", || {});
        let graph = builder.finalize().unwrap();
        assert_eq!(graph.order.len(), 1);
        assert_eq!(graph.order[0], "single");
    }

    #[test]
    fn add_multiple_independent_nodes() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("a", || {});
        builder.add_task("b", || {});
        builder.add_task("c", || {});
        let graph = builder.finalize().unwrap();
        assert_eq!(graph.order.len(), 3);
    }

    #[test]
    fn add_task_overwrites_existing() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut builder = JobGraphBuilder::new();

        let c1 = Arc::clone(&counter);
        builder.add_task("task", move || { c1.store(1, Ordering::SeqCst); });

        let c2 = Arc::clone(&counter);
        builder.add_task("task", move || { c2.store(42, Ordering::SeqCst); });

        let pool = ThreadPool::new(1);
        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        // Second task should have overwritten the first
        assert_eq!(counter.load(Ordering::SeqCst), 42);
        pool.shutdown();
    }

    #[test]
    fn add_edge_creates_dependency() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let pool = ThreadPool::new(2);

        let mut builder = JobGraphBuilder::new();
        let l1 = Arc::clone(&log);
        builder.add_task("first", move || { l1.lock().unwrap().push(1); });
        let l2 = Arc::clone(&log);
        builder.add_task("second", move || { l2.lock().unwrap().push(2); });
        builder.depends_on("first", "second");

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        let entries = log.lock().unwrap().clone();
        assert_eq!(entries.len(), 2);
        let pos_1 = entries.iter().position(|&x| x == 1).unwrap();
        let pos_2 = entries.iter().position(|&x| x == 2).unwrap();
        assert!(pos_1 < pos_2);
        pool.shutdown();
    }

    #[test]
    fn add_multiple_edges_same_source() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("root", || {});
        builder.add_task("child1", || {});
        builder.add_task("child2", || {});
        builder.add_task("child3", || {});
        builder.depends_on("root", "child1");
        builder.depends_on("root", "child2");
        builder.depends_on("root", "child3");

        let graph = builder.finalize().unwrap();
        assert_eq!(graph.order.len(), 4);
        // root should come first
        assert_eq!(graph.order[0], "root");
    }

    #[test]
    fn add_multiple_edges_same_target() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("dep1", || {});
        builder.add_task("dep2", || {});
        builder.add_task("dep3", || {});
        builder.add_task("target", || {});
        builder.depends_on("dep1", "target");
        builder.depends_on("dep2", "target");
        builder.depends_on("dep3", "target");

        let graph = builder.finalize().unwrap();
        assert_eq!(graph.order.len(), 4);
        // target should come last
        assert_eq!(graph.order[3], "target");
    }

    #[test]
    fn builder_default_creates_empty() {
        let builder = JobGraphBuilder::default();
        let graph = builder.finalize().unwrap();
        assert!(graph.order.is_empty());
    }

    #[test]
    fn builder_chaining_works() {
        let mut builder = JobGraphBuilder::new();
        builder
            .add_task("a", || {})
            .add_task("b", || {})
            .add_task("c", || {})
            .depends_on("a", "b")
            .depends_on("b", "c");

        let graph = builder.finalize().unwrap();
        assert_eq!(graph.order.len(), 3);
    }

    #[test]
    fn graph_preserves_indices() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("x", || {});
        builder.add_task("y", || {});
        builder.add_task("z", || {});
        builder.depends_on("x", "y").depends_on("y", "z");

        let graph = builder.finalize().unwrap();

        // Verify indices map correctly
        for (i, name) in graph.order.iter().enumerate() {
            assert_eq!(graph.indices[name], i);
        }
    }

    #[test]
    fn graph_validation_all_tasks_present() {
        let mut builder = JobGraphBuilder::new();
        for i in 0..10 {
            builder.add_task(&format!("task_{}", i), || {});
        }

        let graph = builder.finalize().unwrap();
        assert_eq!(graph.order.len(), 10);
        assert_eq!(graph.indices.len(), 10);
    }

    #[test]
    fn duplicate_edge_handling() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("a", || {});
        builder.add_task("b", || {});
        // Add same edge multiple times
        builder.depends_on("a", "b");
        builder.depends_on("a", "b");
        builder.depends_on("a", "b");

        // Should still finalize without error
        let graph = builder.finalize().unwrap();
        assert_eq!(graph.order.len(), 2);
    }

    // =========================================================================
    // DEPENDENCY RESOLUTION TESTS (10+ tests)
    // =========================================================================

    #[test]
    fn topological_sort_single_node() {
        let names = vec!["solo".to_string()];
        let deps = HashMap::new();
        let dependents = HashMap::new();

        let result = topological_sort(&names, &deps, &dependents).unwrap();
        assert_eq!(result, vec!["solo"]);
    }

    #[test]
    fn topological_sort_linear_chain() {
        let names: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut deps = HashMap::new();
        deps.insert("b".to_string(), vec!["a".to_string()]);
        deps.insert("c".to_string(), vec!["b".to_string()]);
        deps.insert("d".to_string(), vec!["c".to_string()]);

        let mut dependents = HashMap::new();
        dependents.insert("a".to_string(), vec!["b".to_string()]);
        dependents.insert("b".to_string(), vec!["c".to_string()]);
        dependents.insert("c".to_string(), vec!["d".to_string()]);

        let result = topological_sort(&names, &deps, &dependents).unwrap();

        let pos = |s: &str| result.iter().position(|x| x == s).unwrap();
        assert!(pos("a") < pos("b"));
        assert!(pos("b") < pos("c"));
        assert!(pos("c") < pos("d"));
    }

    #[test]
    fn topological_sort_parallel_roots() {
        let names: Vec<String> = vec!["r1", "r2", "r3", "child"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut deps = HashMap::new();
        deps.insert("child".to_string(), vec![
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ]);

        let mut dependents = HashMap::new();
        dependents.insert("r1".to_string(), vec!["child".to_string()]);
        dependents.insert("r2".to_string(), vec!["child".to_string()]);
        dependents.insert("r3".to_string(), vec!["child".to_string()]);

        let result = topological_sort(&names, &deps, &dependents).unwrap();

        // child must come after all roots
        let pos_child = result.iter().position(|x| x == "child").unwrap();
        assert_eq!(pos_child, 3);
    }

    #[test]
    fn cycle_detection_self_loop() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("self", || {});
        builder.depends_on("self", "self");

        match builder.finalize() {
            Err(e) => assert!(e.cycle.contains(&"self".to_string())),
            Ok(_) => panic!("expected cycle error for self-loop"),
        }
    }

    #[test]
    fn cycle_detection_two_node_cycle() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("a", || {});
        builder.add_task("b", || {});
        builder.depends_on("a", "b").depends_on("b", "a");

        match builder.finalize() {
            Err(e) => {
                assert!(e.cycle.contains(&"a".to_string()) ||
                       e.cycle.contains(&"b".to_string()));
            }
            Ok(_) => panic!("expected cycle error"),
        }
    }

    #[test]
    fn cycle_detection_long_cycle() {
        let mut builder = JobGraphBuilder::new();
        for i in 0..10 {
            builder.add_task(&format!("n{}", i), || {});
        }
        for i in 0..9 {
            builder.depends_on(&format!("n{}", i), &format!("n{}", i + 1));
        }
        // Close the cycle
        builder.depends_on("n9", "n0");

        assert!(builder.finalize().is_err());
    }

    #[test]
    fn dependency_levels_correct() {
        let pool = ThreadPool::new(4);
        let level_tracker = Arc::new(Mutex::new(HashMap::new()));
        let current_level = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        // Level 0: roots
        for i in 0..3 {
            let tracker = Arc::clone(&level_tracker);
            let name = format!("root_{}", i);
            let name_clone = name.clone();
            builder.add_task(&name, move || {
                tracker.lock().unwrap().insert(name_clone, 0usize);
            });
        }

        // Level 1: depends on roots
        for i in 0..3 {
            let tracker = Arc::clone(&level_tracker);
            let name = format!("mid_{}", i);
            let name_clone = name.clone();
            builder.add_task(&name, move || {
                tracker.lock().unwrap().insert(name_clone, 1usize);
            });
            builder.depends_on(&format!("root_{}", i), &format!("mid_{}", i));
        }

        // Level 2: depends on mids
        let tracker = Arc::clone(&level_tracker);
        builder.add_task("final", move || {
            tracker.lock().unwrap().insert("final".to_string(), 2usize);
        });
        for i in 0..3 {
            builder.depends_on(&format!("mid_{}", i), "final");
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        let levels = level_tracker.lock().unwrap();
        assert_eq!(*levels.get("root_0").unwrap(), 0);
        assert_eq!(*levels.get("mid_0").unwrap(), 1);
        assert_eq!(*levels.get("final").unwrap(), 2);

        pool.shutdown();
    }

    #[test]
    fn parallel_groups_identified() {
        let pool = ThreadPool::new(8);
        let execution_times = Arc::new(Mutex::new(Vec::new()));

        let mut builder = JobGraphBuilder::new();

        // All independent tasks - should run in parallel
        for i in 0..5 {
            let times = Arc::clone(&execution_times);
            builder.add_task(&format!("parallel_{}", i), move || {
                let start = std::time::Instant::now();
                std::thread::sleep(std::time::Duration::from_millis(10));
                times.lock().unwrap().push(start);
            });
        }

        let graph = builder.finalize().unwrap();
        let overall_start = std::time::Instant::now();
        graph.execute(&pool);
        let overall_duration = overall_start.elapsed();

        // If truly parallel, total time should be ~10ms not ~50ms
        // Allow some slack for scheduling
        assert!(overall_duration < std::time::Duration::from_millis(40));

        pool.shutdown();
    }

    #[test]
    fn unknown_from_dependency_rejected() {
        let mut builder = JobGraphBuilder::new();
        builder.add_task("real", || {});
        builder.depends_on("fake", "real");

        match builder.finalize() {
            Err(e) => assert!(e.cycle[0].contains("fake")),
            Ok(_) => panic!("expected error for unknown 'from' task"),
        }
    }

    #[test]
    fn complex_dag_validates() {
        let mut builder = JobGraphBuilder::new();

        //       a
        //      / \
        //     b   c
        //    / \ / \
        //   d   e   f
        //    \ | /
        //      g

        builder.add_task("a", || {});
        builder.add_task("b", || {});
        builder.add_task("c", || {});
        builder.add_task("d", || {});
        builder.add_task("e", || {});
        builder.add_task("f", || {});
        builder.add_task("g", || {});

        builder.depends_on("a", "b").depends_on("a", "c");
        builder.depends_on("b", "d").depends_on("b", "e");
        builder.depends_on("c", "e").depends_on("c", "f");
        builder.depends_on("d", "g").depends_on("e", "g").depends_on("f", "g");

        let graph = builder.finalize().unwrap();
        assert_eq!(graph.order.len(), 7);

        // Verify ordering constraints
        let pos = |s: &str| graph.order.iter().position(|x| x == s).unwrap();
        assert!(pos("a") < pos("b"));
        assert!(pos("a") < pos("c"));
        assert!(pos("b") < pos("d"));
        assert!(pos("b") < pos("e"));
        assert!(pos("c") < pos("e"));
        assert!(pos("c") < pos("f"));
        assert!(pos("d") < pos("g"));
        assert!(pos("e") < pos("g"));
        assert!(pos("f") < pos("g"));
    }

    // =========================================================================
    // EXECUTION TESTS (10+ tests)
    // =========================================================================

    #[test]
    fn execution_order_sequential() {
        let pool = ThreadPool::new(1); // Single thread forces sequential
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut builder = JobGraphBuilder::new();
        for i in 0..5 {
            let l = Arc::clone(&log);
            builder.add_task(&format!("t{}", i), move || {
                l.lock().unwrap().push(i);
            });
            if i > 0 {
                builder.depends_on(&format!("t{}", i - 1), &format!("t{}", i));
            }
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        let entries = log.lock().unwrap().clone();
        assert_eq!(entries, vec![0, 1, 2, 3, 4]);

        pool.shutdown();
    }

    #[test]
    fn execution_all_tasks_complete() {
        let pool = ThreadPool::new(4);
        let completed = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();
        for i in 0..20 {
            let c = Arc::clone(&completed);
            builder.add_task(&format!("task_{}", i), move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        let graph = builder.finalize().unwrap();
        let handles = graph.execute(&pool);

        assert_eq!(handles.len(), 20);
        assert_eq!(completed.load(Ordering::SeqCst), 20);

        // All handles should report complete
        for (_, handle) in &handles {
            assert!(handle.is_complete());
        }

        pool.shutdown();
    }

    #[test]
    fn execution_respects_dependencies() {
        let pool = ThreadPool::new(8);
        let value = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        let v1 = Arc::clone(&value);
        builder.add_task("set_10", move || {
            v1.store(10, Ordering::SeqCst);
        });

        let v2 = Arc::clone(&value);
        builder.add_task("multiply_2", move || {
            let current = v2.load(Ordering::SeqCst);
            v2.store(current * 2, Ordering::SeqCst);
        });

        let v3 = Arc::clone(&value);
        builder.add_task("add_5", move || {
            let current = v3.load(Ordering::SeqCst);
            v3.store(current + 5, Ordering::SeqCst);
        });

        builder.depends_on("set_10", "multiply_2");
        builder.depends_on("multiply_2", "add_5");

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        // 10 -> 20 -> 25
        assert_eq!(value.load(Ordering::SeqCst), 25);

        pool.shutdown();
    }

    #[test]
    fn execution_handles_returned() {
        let pool = ThreadPool::new(2);

        let mut builder = JobGraphBuilder::new();
        builder.add_task("one", || {});
        builder.add_task("two", || {});
        builder.add_task("three", || {});

        let graph = builder.finalize().unwrap();
        let handles = graph.execute(&pool);

        assert!(handles.contains_key("one"));
        assert!(handles.contains_key("two"));
        assert!(handles.contains_key("three"));

        pool.shutdown();
    }

    #[test]
    fn execution_parallel_tasks_interleave() {
        let pool = ThreadPool::new(4);
        let order = Arc::new(Mutex::new(Vec::new()));

        let mut builder = JobGraphBuilder::new();

        // Root task
        let o1 = Arc::clone(&order);
        builder.add_task("root", move || {
            o1.lock().unwrap().push("root_start");
            o1.lock().unwrap().push("root_end");
        });

        // Parallel children
        for i in 0..3 {
            let o = Arc::clone(&order);
            let name = format!("child_{}", i);
            let name_copy = name.clone();
            builder.add_task(&name, move || {
                o.lock().unwrap().push(Box::leak(format!("{}_exec", name_copy).into_boxed_str()));
            });
            builder.depends_on("root", &format!("child_{}", i));
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        let entries = order.lock().unwrap().clone();
        // Root must complete before children start
        let root_end_pos = entries.iter().position(|&x| x == "root_end").unwrap();
        for entry in entries.iter().skip(root_end_pos + 1) {
            assert!(entry.contains("child"));
        }

        pool.shutdown();
    }

    #[test]
    fn execution_completion_tracking_accurate() {
        let pool = ThreadPool::new(2);

        let mut builder = JobGraphBuilder::new();
        builder.add_task("slow", || {
            std::thread::sleep(std::time::Duration::from_millis(20));
        });
        builder.add_task("fast", || {});
        builder.depends_on("slow", "fast");

        let graph = builder.finalize().unwrap();
        let handles = graph.execute(&pool);

        // Both should be complete after execute returns (blocking)
        assert!(handles["slow"].is_complete());
        assert!(handles["fast"].is_complete());

        pool.shutdown();
    }

    #[test]
    fn execution_with_varying_pool_sizes() {
        for pool_size in [1, 2, 4, 8] {
            let pool = ThreadPool::new(pool_size);
            let sum = Arc::new(AtomicUsize::new(0));

            let mut builder = JobGraphBuilder::new();
            for i in 1..=10 {
                let s = Arc::clone(&sum);
                builder.add_task(&format!("add_{}", i), move || {
                    s.fetch_add(i, Ordering::SeqCst);
                });
            }

            let graph = builder.finalize().unwrap();
            graph.execute(&pool);

            // Sum of 1..=10 = 55
            assert_eq!(sum.load(Ordering::SeqCst), 55);

            pool.shutdown();
        }
    }

    #[test]
    fn execution_many_dependencies_single_target() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        // 10 tasks that each increment counter
        for i in 0..10 {
            let c = Arc::clone(&counter);
            builder.add_task(&format!("dep_{}", i), move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        // Final task checks all completed
        let c = Arc::clone(&counter);
        builder.add_task("final", move || {
            assert_eq!(c.load(Ordering::SeqCst), 10);
        });

        for i in 0..10 {
            builder.depends_on(&format!("dep_{}", i), "final");
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        pool.shutdown();
    }

    #[test]
    fn execution_single_task_graph() {
        let pool = ThreadPool::new(2);
        let executed = Arc::new(AtomicBool::new(false));

        let mut builder = JobGraphBuilder::new();
        let e = Arc::clone(&executed);
        builder.add_task("only", move || {
            e.store(true, Ordering::SeqCst);
        });

        let graph = builder.finalize().unwrap();
        let handles = graph.execute(&pool);

        assert!(executed.load(Ordering::SeqCst));
        assert!(handles["only"].is_complete());

        pool.shutdown();
    }

    #[test]
    fn execution_deeply_nested_dependencies() {
        let pool = ThreadPool::new(2);
        let value = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        // Chain of 20 tasks, each incrementing
        for i in 0..20 {
            let v = Arc::clone(&value);
            builder.add_task(&format!("step_{}", i), move || {
                v.fetch_add(1, Ordering::SeqCst);
            });
            if i > 0 {
                builder.depends_on(&format!("step_{}", i - 1), &format!("step_{}", i));
            }
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert_eq!(value.load(Ordering::SeqCst), 20);

        pool.shutdown();
    }

    // =========================================================================
    // EDGE CASE TESTS (10+ tests)
    // =========================================================================

    #[test]
    fn edge_case_single_isolated_node() {
        let pool = ThreadPool::new(1);
        let ran = Arc::new(AtomicBool::new(false));

        let mut builder = JobGraphBuilder::new();
        let r = Arc::clone(&ran);
        builder.add_task("isolated", move || {
            r.store(true, Ordering::SeqCst);
        });

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert!(ran.load(Ordering::SeqCst));
        pool.shutdown();
    }

    #[test]
    fn edge_case_diamond_with_extra_edges() {
        let pool = ThreadPool::new(4);
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut builder = JobGraphBuilder::new();

        //     a
        //    /|\
        //   b c d
        //    \|/
        //     e
        //     |
        //     f

        for name in ["a", "b", "c", "d", "e", "f"] {
            let l = Arc::clone(&log);
            let n = name.to_string();
            builder.add_task(name, move || {
                l.lock().unwrap().push(n);
            });
        }

        builder.depends_on("a", "b").depends_on("a", "c").depends_on("a", "d");
        builder.depends_on("b", "e").depends_on("c", "e").depends_on("d", "e");
        builder.depends_on("e", "f");

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        let entries = log.lock().unwrap().clone();
        let pos = |s: &str| entries.iter().position(|x| x == s).unwrap();

        assert!(pos("a") < pos("b"));
        assert!(pos("a") < pos("c"));
        assert!(pos("a") < pos("d"));
        assert!(pos("b") < pos("e"));
        assert!(pos("c") < pos("e"));
        assert!(pos("d") < pos("e"));
        assert!(pos("e") < pos("f"));

        pool.shutdown();
    }

    #[test]
    fn edge_case_long_chain_100_nodes() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        for i in 0..100 {
            let c = Arc::clone(&counter);
            builder.add_task(&format!("n{}", i), move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
            if i > 0 {
                builder.depends_on(&format!("n{}", i - 1), &format!("n{}", i));
            }
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert_eq!(counter.load(Ordering::SeqCst), 100);
        pool.shutdown();
    }

    #[test]
    fn edge_case_wide_graph_50_parallel() {
        let pool = ThreadPool::new(8);
        let counter = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        builder.add_task("root", || {});

        for i in 0..50 {
            let c = Arc::clone(&counter);
            builder.add_task(&format!("parallel_{}", i), move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
            builder.depends_on("root", &format!("parallel_{}", i));
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert_eq!(counter.load(Ordering::SeqCst), 50);
        pool.shutdown();
    }

    #[test]
    fn edge_case_multiple_roots() {
        let pool = ThreadPool::new(4);
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut builder = JobGraphBuilder::new();

        // Three independent roots
        for i in 0..3 {
            let l = Arc::clone(&log);
            builder.add_task(&format!("root_{}", i), move || {
                l.lock().unwrap().push(format!("root_{}", i));
            });
        }

        // Single sink depending on all
        let l = Arc::clone(&log);
        builder.add_task("sink", move || {
            l.lock().unwrap().push("sink".to_string());
        });

        for i in 0..3 {
            builder.depends_on(&format!("root_{}", i), "sink");
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        let entries = log.lock().unwrap().clone();
        let sink_pos = entries.iter().position(|x| x == "sink").unwrap();
        assert_eq!(sink_pos, 3); // sink must be last

        pool.shutdown();
    }

    #[test]
    fn edge_case_multiple_sinks() {
        let pool = ThreadPool::new(4);
        let executed = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        // Single root
        let e = Arc::clone(&executed);
        builder.add_task("root", move || {
            e.fetch_add(1, Ordering::SeqCst);
        });

        // Multiple independent sinks
        for i in 0..5 {
            let e = Arc::clone(&executed);
            builder.add_task(&format!("sink_{}", i), move || {
                e.fetch_add(1, Ordering::SeqCst);
            });
            builder.depends_on("root", &format!("sink_{}", i));
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert_eq!(executed.load(Ordering::SeqCst), 6);
        pool.shutdown();
    }

    #[test]
    fn edge_case_binary_tree_structure() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        // Build binary tree: depth 4 = 15 nodes
        // Level 0: node_0
        // Level 1: node_1, node_2
        // Level 2: node_3, node_4, node_5, node_6
        // Level 3: node_7..node_14

        for i in 0..15 {
            let c = Arc::clone(&counter);
            builder.add_task(&format!("node_{}", i), move || {
                c.fetch_add(1, Ordering::SeqCst);
            });

            if i > 0 {
                let parent = (i - 1) / 2;
                builder.depends_on(&format!("node_{}", parent), &format!("node_{}", i));
            }
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert_eq!(counter.load(Ordering::SeqCst), 15);
        pool.shutdown();
    }

    #[test]
    fn edge_case_task_with_closure_capture() {
        let pool = ThreadPool::new(2);

        let shared_data = Arc::new(Mutex::new(vec![1, 2, 3]));
        let result = Arc::new(Mutex::new(0));

        let mut builder = JobGraphBuilder::new();

        let data = Arc::clone(&shared_data);
        let res = Arc::clone(&result);
        builder.add_task("sum", move || {
            let sum: i32 = data.lock().unwrap().iter().sum();
            *res.lock().unwrap() = sum;
        });

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert_eq!(*result.lock().unwrap(), 6);
        pool.shutdown();
    }

    #[test]
    fn edge_case_cycle_error_display() {
        let err = CycleError {
            cycle: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        };

        let msg = format!("{}", err);
        assert!(msg.contains("a"));
        assert!(msg.contains("b"));
        assert!(msg.contains("c"));
        assert!(msg.contains("->"));
    }

    #[test]
    fn edge_case_cycle_error_is_error_trait() {
        let err = CycleError {
            cycle: vec!["test".to_string()],
        };

        // Verify Error trait is implemented
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn edge_case_task_handle_wait_immediate() {
        let done = Arc::new(AtomicBool::new(true));
        let handle = TaskHandle {
            done: Arc::clone(&done),
        };

        // Should return immediately since already complete
        handle.wait();
        assert!(handle.is_complete());
    }

    #[test]
    fn edge_case_interleaved_add_and_depends() {
        let pool = ThreadPool::new(2);
        let log = Arc::new(Mutex::new(Vec::new()));

        let mut builder = JobGraphBuilder::new();

        let l1 = Arc::clone(&log);
        builder.add_task("a", move || { l1.lock().unwrap().push("a"); });

        // Add dependency before adding the target task
        builder.depends_on("a", "b");

        let l2 = Arc::clone(&log);
        builder.add_task("b", move || { l2.lock().unwrap().push("b"); });

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        let entries = log.lock().unwrap().clone();
        let pos_a = entries.iter().position(|x| *x == "a").unwrap();
        let pos_b = entries.iter().position(|x| *x == "b").unwrap();
        assert!(pos_a < pos_b);

        pool.shutdown();
    }

    // =========================================================================
    // ADDITIONAL STRESS AND CORRECTNESS TESTS
    // =========================================================================

    #[test]
    fn stress_many_small_tasks() {
        let pool = ThreadPool::new(8);
        let counter = Arc::new(AtomicUsize::new(0));

        let mut builder = JobGraphBuilder::new();

        for i in 0..200 {
            let c = Arc::clone(&counter);
            builder.add_task(&format!("task_{}", i), move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        assert_eq!(counter.load(Ordering::SeqCst), 200);
        pool.shutdown();
    }

    #[test]
    fn correctness_all_handles_populated() {
        let pool = ThreadPool::new(4);

        let mut builder = JobGraphBuilder::new();
        let names: Vec<String> = (0..25).map(|i| format!("task_{}", i)).collect();

        for name in &names {
            builder.add_task(name, || {});
        }

        let graph = builder.finalize().unwrap();
        let handles = graph.execute(&pool);

        for name in &names {
            assert!(handles.contains_key(name), "Missing handle for {}", name);
            assert!(handles[name].is_complete());
        }

        pool.shutdown();
    }

    #[test]
    fn correctness_order_vector_matches_indices() {
        let mut builder = JobGraphBuilder::new();

        for i in 0..10 {
            builder.add_task(&format!("t{}", i), || {});
        }

        // Create some dependencies
        builder.depends_on("t0", "t5");
        builder.depends_on("t1", "t5");
        builder.depends_on("t5", "t9");

        let graph = builder.finalize().unwrap();

        // Verify order and indices are consistent
        for (idx, name) in graph.order.iter().enumerate() {
            assert_eq!(
                graph.indices.get(name).copied(),
                Some(idx),
                "Index mismatch for {}",
                name
            );
        }
    }

    #[test]
    fn correctness_no_task_runs_twice() {
        let pool = ThreadPool::new(4);
        let counters: Vec<Arc<AtomicUsize>> = (0..10)
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect();

        let mut builder = JobGraphBuilder::new();

        for (i, counter) in counters.iter().enumerate() {
            let c = Arc::clone(counter);
            builder.add_task(&format!("task_{}", i), move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        // Create some dependencies
        builder.depends_on("task_0", "task_5");
        builder.depends_on("task_1", "task_5");
        builder.depends_on("task_5", "task_9");

        let graph = builder.finalize().unwrap();
        graph.execute(&pool);

        // Each task should have run exactly once
        for (i, counter) in counters.iter().enumerate() {
            assert_eq!(
                counter.load(Ordering::SeqCst),
                1,
                "Task {} ran {} times",
                i,
                counter.load(Ordering::SeqCst)
            );
        }

        pool.shutdown();
    }
}
