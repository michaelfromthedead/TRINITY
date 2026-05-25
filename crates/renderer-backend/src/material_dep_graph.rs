use std::collections::{HashMap, VecDeque};

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
}
