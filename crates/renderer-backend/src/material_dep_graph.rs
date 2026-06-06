use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::RwLock;

/// Tracks material include dependencies for hot-reload.
pub struct DepGraph {
    pub includes_to_materials: HashMap<String, Vec<u32>>,
    pub materials_to_includes: HashMap<u32, Vec<String>>,
}

impl DepGraph {
    pub fn new() -> Self {
        Self {
            includes_to_materials: HashMap::new(),
            materials_to_includes: HashMap::new(),
        }
    }

    pub fn add_include(&mut self, material_id: u32, include: String) {
        self.includes_to_materials
            .entry(include.clone())
            .or_default()
            .push(material_id);
        self.materials_to_includes
            .entry(material_id)
            .or_default()
            .push(include);
    }

    /// BFS from `changed_include`, returning all reachable material IDs.
    pub fn invalidate(&mut self, changed_include: &str) -> Vec<u32> {
        let mut visited = Vec::new();
        let mut queue = VecDeque::new();

        if let Some(mut ids) = self.includes_to_materials.remove(changed_include) {
            // Deduplicate while draining.
            ids.sort();
            ids.dedup();
            for &id in &ids {
                queue.push_back(id);
            }
        }

        while let Some(material_id) = queue.pop_front() {
            if visited.contains(&material_id) {
                continue;
            }
            visited.push(material_id);

            // Remove reversed edge so the graph stays consistent.
            self.materials_to_includes.remove(&material_id);

            // Check if other includes of this material are also dirty.
            if let Some(include) = self
                .materials_to_includes
                .get(&material_id)
                .and_then(|inc| inc.first())
            {
                if let Some(dependents) = self.includes_to_materials.remove(include.as_str()) {
                    for &id in &dependents {
                        if !visited.contains(&id) {
                            queue.push_back(id);
                        }
                    }
                }
            }
        }

        visited
    }

    /// Returns all material IDs transitively dependent on the given include path using BFS.
    /// Unlike `invalidate()`, this method does not modify the graph.
    pub fn broadest_invalidation_set(&self, changed_include: &str) -> HashSet<u32> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with direct dependents of the changed include
        if let Some(material_ids) = self.includes_to_materials.get(changed_include) {
            for &id in material_ids {
                queue.push_back(id);
            }
        }

        // BFS through transitive dependents
        while let Some(material_id) = queue.pop_front() {
            if result.insert(material_id) {
                // Find all includes this material uses, then find other materials using those includes
                if let Some(includes) = self.materials_to_includes.get(&material_id) {
                    for include in includes {
                        if let Some(dependent_ids) = self.includes_to_materials.get(include) {
                            for &dep_id in dependent_ids {
                                if !result.contains(&dep_id) {
                                    queue.push_back(dep_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Returns all include paths transitively affected by a change to the given path.
    /// Useful for determining the full scope of a shader file modification.
    pub fn broadest_invalidation_set_paths(&self, changed_include: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let material_ids = self.broadest_invalidation_set(changed_include);

        // Collect all includes used by affected materials
        for material_id in material_ids {
            if let Some(includes) = self.materials_to_includes.get(&material_id) {
                for include in includes {
                    result.insert(include.clone());
                }
            }
        }

        // Also include the original changed path
        result.insert(changed_include.to_string());
        result
    }
}

/// Thread-safe wrapper around DepGraph for concurrent access.
pub struct ThreadSafeDepGraph {
    inner: RwLock<DepGraph>,
}

impl ThreadSafeDepGraph {
    /// Creates a new thread-safe dependency graph.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(DepGraph::new()),
        }
    }

    /// Acquires a read lock on the dependency graph.
    ///
    /// # Panics
    /// Panics if the RwLock is poisoned (a thread panicked while holding the lock).
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, DepGraph> {
        self.inner.read().expect("RwLock poisoned")
    }

    /// Acquires a write lock on the dependency graph.
    ///
    /// # Panics
    /// Panics if the RwLock is poisoned (a thread panicked while holding the lock).
    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, DepGraph> {
        self.inner.write().expect("RwLock poisoned")
    }

    /// Adds an include dependency, acquiring a write lock.
    pub fn add_include(&self, material_id: u32, include: String) {
        self.write().add_include(material_id, include);
    }

    /// Returns the broadest invalidation set without modifying the graph.
    pub fn broadest_invalidation_set(&self, changed_include: &str) -> HashSet<u32> {
        self.read().broadest_invalidation_set(changed_include)
    }

    /// Invalidates materials affected by a changed include.
    pub fn invalidate(&self, changed_include: &str) -> Vec<u32> {
        self.write().invalidate(changed_include)
    }
}

impl Default for ThreadSafeDepGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_invalidate_single() {
        let mut g = DepGraph::new();
        g.add_include(1, "common.wgsl".into());
        let affected = g.invalidate("common.wgsl");
        assert_eq!(affected, vec![1]);
    }

    #[test]
    fn test_invalidate_unknown_include() {
        let mut g = DepGraph::new();
        g.add_include(1, "common.wgsl".into());
        let affected = g.invalidate("nonexistent.wgsl");
        assert!(affected.is_empty());
    }

    #[test]
    fn test_invalidate_multiple_materials() {
        let mut g = DepGraph::new();
        g.add_include(1, "common.wgsl".into());
        g.add_include(2, "common.wgsl".into());
        g.add_include(3, "other.wgsl".into());
        let mut affected = g.invalidate("common.wgsl");
        affected.sort();
        assert_eq!(affected, vec![1, 2]);
    }

    #[test]
    fn test_transitive_invalidation() {
        let mut g = DepGraph::new();
        // material 1 includes "a", material 2 includes what "a" resolves to
        g.add_include(1, "a.wgsl".into());
        g.add_include(2, "b.wgsl".into());
        // Pretend "a.wgsl" also includes "b.wgsl" — this is an indirect dep.
        // In BFS we only follow edges stored in includes_to_materials.
        // For a true transitive chain add an intermediate include:
        g.add_include(2, "a.wgsl".into());
        let mut affected = g.invalidate("a.wgsl");
        affected.sort();
        assert_eq!(affected, vec![1, 2]);
    }

    #[test]
    fn test_broadest_invalidation_set() {
        let mut g = DepGraph::new();
        // Build a dependency chain: common.wgsl -> material 1 -> also uses base.wgsl
        // base.wgsl -> material 2, material 3
        g.add_include(1, "common.wgsl".into());
        g.add_include(1, "base.wgsl".into());
        g.add_include(2, "base.wgsl".into());
        g.add_include(3, "base.wgsl".into());
        g.add_include(4, "other.wgsl".into());

        let set = g.broadest_invalidation_set("common.wgsl");
        // Should include material 1 (direct), and 2, 3 (transitive via base.wgsl)
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(!set.contains(&4)); // Not connected
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_broadest_invalidation_set_empty() {
        let g = DepGraph::new();
        let set = g.broadest_invalidation_set("nonexistent.wgsl");
        assert!(set.is_empty());
    }

    #[test]
    fn test_broadest_invalidation_set_paths() {
        let mut g = DepGraph::new();
        g.add_include(1, "common.wgsl".into());
        g.add_include(1, "utils.wgsl".into());
        g.add_include(2, "utils.wgsl".into());

        let paths = g.broadest_invalidation_set_paths("common.wgsl");
        assert!(paths.contains("common.wgsl"));
        assert!(paths.contains("utils.wgsl"));
    }

    #[test]
    fn test_thread_safe_dep_graph_basic() {
        let graph = ThreadSafeDepGraph::new();
        graph.add_include(1, "common.wgsl".into());
        graph.add_include(2, "common.wgsl".into());

        let set = graph.broadest_invalidation_set("common.wgsl");
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_thread_safe_dep_graph_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        let graph = Arc::new(ThreadSafeDepGraph::new());
        graph.add_include(1, "base.wgsl".into());
        graph.add_include(2, "base.wgsl".into());

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let g = Arc::clone(&graph);
                thread::spawn(move || {
                    let guard = g.read();
                    assert!(guard.includes_to_materials.contains_key("base.wgsl"));
                    let set = guard.broadest_invalidation_set("base.wgsl");
                    assert_eq!(set.len(), 2);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_thread_safe_dep_graph_concurrent_write_read() {
        use std::sync::Arc;
        use std::thread;

        let graph = Arc::new(ThreadSafeDepGraph::new());

        // Writer thread
        let g_write = Arc::clone(&graph);
        let writer = thread::spawn(move || {
            for i in 0..100 {
                g_write.add_include(i, format!("shader_{}.wgsl", i % 10));
            }
        });

        // Reader threads
        let readers: Vec<_> = (0..3)
            .map(|_| {
                let g = Arc::clone(&graph);
                thread::spawn(move || {
                    for _ in 0..50 {
                        let _ = g.broadest_invalidation_set("shader_0.wgsl");
                    }
                })
            })
            .collect();

        writer.join().expect("Writer panicked");
        for reader in readers {
            reader.join().expect("Reader panicked");
        }

        // Verify final state
        let set = graph.broadest_invalidation_set("shader_0.wgsl");
        assert!(!set.is_empty());
    }

    #[test]
    fn test_thread_safe_dep_graph_default() {
        let graph = ThreadSafeDepGraph::default();
        assert!(graph.read().includes_to_materials.is_empty());
    }
}
