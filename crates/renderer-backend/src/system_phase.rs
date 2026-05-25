//! Ordered system execution with dependency-aware phase graphs.
//!
//! Provides [`SystemContext`] (per-frame timing and metadata), the [`System`]
//! trait for frame-level update logic, [`SystemPhase`] (an ordered collection
//! of systems that run sequentially), and [`PhaseGraph`] (a DAG of phases
//! with dependency-based topological execution).
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::system_phase::{System, SystemContext, SystemPhase, PhaseGraph};
//!
//! struct Physics;
//! impl System for Physics {
//!     fn run(&mut self, ctx: &SystemContext) {
//!         // advance simulation by ctx.delta_time
//!     }
//! }
//!
//! let mut graph = PhaseGraph::new();
//! graph.add_phase(SystemPhase::new("simulation"));
//! graph.phase_mut("simulation").add_system(Physics);
//! graph.add_dependency("rendering", "simulation");
//! graph.execute(&SystemContext::new(1.0 / 60.0, 1));
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// SystemContext
// ---------------------------------------------------------------------------

/// Per-frame context passed to every [`System::run`] invocation.
#[derive(Clone, Debug)]
pub struct SystemContext {
    /// Time delta in seconds since the last frame.
    pub delta_time: f32,
    /// Monotonically increasing frame counter.
    pub frame_number: u64,
}

impl SystemContext {
    /// Creates a new context for the given frame parameters.
    pub const fn new(delta_time: f32, frame_number: u64) -> Self {
        Self {
            delta_time,
            frame_number,
        }
    }
}

// ---------------------------------------------------------------------------
// System trait
// ---------------------------------------------------------------------------

/// A single updateable system within a render pipeline phase.
///
/// Implementors receive a shared [`SystemContext`] each frame and should
/// mutate their own internal state in response.
pub trait System {
    /// Advance the system by one frame.
    fn run(&mut self, ctx: &SystemContext);
}

// Blanket impl for closures: `FnMut(&SystemContext) -> ()` can be used as a
// System without wrapping in a named struct.
impl<F> System for F
where
    F: FnMut(&SystemContext),
{
    fn run(&mut self, ctx: &SystemContext) {
        (self)(ctx);
    }
}

// ---------------------------------------------------------------------------
// SystemPhase
// ---------------------------------------------------------------------------

/// An ordered, named collection of systems that execute sequentially.
///
/// Each phase has an `enabled` flag that the [`PhaseGraph`] checks before
/// execution. Disabled phases are skipped during topological traversal.
pub struct SystemPhase {
    /// Debug / friendly name (used as a key by [`PhaseGraph`]).
    name: String,
    /// Whether this phase should be executed.
    pub enabled: bool,
    /// Systems in declaration order.
    systems: Vec<Box<dyn System + Send>>,
}

impl SystemPhase {
    /// Creates a new, enabled phase with the given name and no systems.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            systems: Vec::new(),
        }
    }

    /// Appends a system to the end of this phase's execution order.
    pub fn add_system<S: System + Send + 'static>(&mut self, system: S) {
        self.systems.push(Box::new(system));
    }

    /// Returns the phase name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Runs every system in declaration order.
    ///
    /// This is a no-op when `self.enabled` is `false`.
    pub fn run_all(&mut self, ctx: &SystemContext) {
        if !self.enabled {
            return;
        }
        for s in &mut self.systems {
            s.run(ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// PhaseGraph
// ---------------------------------------------------------------------------

/// A directed acyclic graph of named [`SystemPhase`] values.
///
/// Phases declare dependencies on one another; the graph computes a
/// topological ordering and executes phases in that order, skipping any
/// disabled phases.
pub struct PhaseGraph {
    /// Phases keyed by name.
    phases: HashMap<String, SystemPhase>,
    /// Directed edges: `(phase, depends_on)` — the phase runs *after* its
    /// dependency.
    edges: Vec<(String, String)>,
}

impl PhaseGraph {
    /// Creates an empty phase graph.
    pub fn new() -> Self {
        Self {
            phases: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Inserts a phase into the graph, replacing any existing phase with the
    /// same name.
    pub fn add_phase(&mut self, phase: SystemPhase) {
        let name = phase.name.clone();
        self.phases.insert(name, phase);
    }

    /// Returns a mutable reference to a phase by name, or `None`.
    pub fn phase_mut(&mut self, name: &str) -> Option<&mut SystemPhase> {
        self.phases.get_mut(name)
    }

    /// Returns a shared reference to a phase by name, or `None`.
    pub fn phase(&self, name: &str) -> Option<&SystemPhase> {
        self.phases.get(name)
    }

    /// Declares that `phase` depends on `depends_on`.
    ///
    /// During execution, `depends_on` will run before `phase`.  Duplicate
    /// edges are silently ignored.
    ///
    /// # Panics
    ///
    /// Panics if either name has not been registered via [`add_phase`].
    ///
    /// [`add_phase`]: Self::add_phase
    pub fn add_dependency(&mut self, phase: &str, depends_on: &str) {
        assert!(
            self.phases.contains_key(phase),
            "PhaseGraph::add_dependency: unknown phase '{phase}'"
        );
        assert!(
            self.phases.contains_key(depends_on),
            "PhaseGraph::add_dependency: unknown dependency '{depends_on}'"
        );
        let edge = (phase.to_string(), depends_on.to_string());
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
    }

    /// Returns `true` when the graph contains no phases.
    pub fn is_empty(&self) -> bool {
        self.phases.is_empty()
    }

    /// Returns the number of registered phases.
    pub fn len(&self) -> usize {
        self.phases.len()
    }

    /// Remove a phase by name, returning it if present.
    ///
    /// Edge entries referencing this name are left untouched.
    pub fn remove_phase(&mut self, name: &str) -> Option<SystemPhase> {
        self.phases.remove(name)
    }

    /// Re-insert a phase that was previously removed (e.g. after
    /// out-of-band execution).
    pub fn insert_phase(&mut self, phase: SystemPhase) {
        let name = phase.name.clone();
        self.phases.insert(name, phase);
    }

    /// Iterate over all dependency edges `(phase, depends_on)`.
    pub fn edges(&self) -> impl Iterator<Item = &(String, String)> {
        self.edges.iter()
    }

    /// Computes the topological order of **enabled** phases using Kahn's
    /// algorithm.
    ///
    /// Returns owned [`String`] values so the result does not borrow `self`,
    /// allowing the caller to mutate the graph after ordering.
    pub fn topological_order(&self) -> Result<Vec<String>, String> {
        // Collect only enabled phases.
        let enabled: HashSet<String> = self
            .phases
            .values()
            .filter(|p| p.enabled)
            .map(|p| p.name.clone())
            .collect();

        if enabled.is_empty() {
            return Ok(Vec::new());
        }

        // Build in-degree map and adjacency list (inverted direction: edges
        // are (phase -> depends_on), so we walk the inverse for Kahn).
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

        for name in &enabled {
            in_degree.insert(name.clone(), 0);
            dependents.insert(name.clone(), Vec::new());
        }

        for (phase, dep) in &self.edges {
            if !enabled.contains(phase) || !enabled.contains(dep) {
                continue;
            }
            // dep must run before phase -> dep has an outgoing edge to phase
            dependents.get_mut(dep).unwrap().push(phase.clone());
            *in_degree.get_mut(phase).unwrap() += 1;
        }

        // Seed queue with phases that have no dependencies.
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(name, _)| name.clone())
            .collect();

        // Sort for deterministic ordering (ties broken by name).
        let mut sorted: Vec<String> = Vec::with_capacity(enabled.len());
        let mut remaining = enabled.len();

        let mut temp: Vec<String> = Vec::new();
        while let Some(current) = queue.pop_front() {
            sorted.push(current.clone());
            remaining -= 1;

            if let Some(deps) = dependents.get(&current) {
                for next in deps {
                    if let Some(deg) = in_degree.get_mut(next) {
                        *deg -= 1;
                        if *deg == 0 {
                            temp.push(next.clone());
                        }
                    }
                }
            }

            if queue.is_empty() && !temp.is_empty() {
                temp.sort();
                queue.extend(temp.drain(..));
            }
        }

        if remaining > 0 {
            let cycle_nodes: Vec<String> = in_degree
                .into_iter()
                .filter(|(_, deg)| *deg > 0)
                .map(|(name, _)| name)
                .collect();
            return Err(format!(
                "PhaseGraph cycle detected involving {} phase(s): {:?}",
                cycle_nodes.len(),
                cycle_nodes,
            ));
        }

        Ok(sorted)
    }

    /// Executes all enabled phases in topologically-determined order.
    ///
    /// Returns `Ok(())` on success, or `Err` describing a cycle that
    /// prevented execution.
    pub fn execute(&mut self, ctx: &SystemContext) -> Result<(), String> {
        let order = self.topological_order()?;
        for name in &order {
            if let Some(phase) = self.phases.get_mut(name.as_str()) {
                phase.run_all(ctx);
            }
        }
        Ok(())
    }
}

impl Default for PhaseGraph {
    fn default() -> Self {
        Self::new()
    }
}
