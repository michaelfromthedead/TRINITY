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
}
