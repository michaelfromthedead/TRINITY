/**
 * Graph Algorithms
 *
 * Algorithms for working with node graphs, including
 * topological sorting, cycle detection, and path finding.
 */

import type { EntityId } from '../types/primitives.js';
import type { WorkflowSchema } from '../types/workflow.js';

// =============================================================================
// GRAPH REPRESENTATION
// =============================================================================

/**
 * Adjacency list representation of a graph.
 */
export type AdjacencyList = ReadonlyMap<EntityId, readonly EntityId[]>;

/**
 * Build an adjacency list from workflow links.
 * Direction: source -> target (following data flow).
 */
export function buildAdjacencyList(workflow: WorkflowSchema): AdjacencyList {
  const adjacency = new Map<EntityId, EntityId[]>();

  // Initialize all nodes with empty arrays
  for (const nodeId of Object.keys(workflow.nodes)) {
    adjacency.set(nodeId, []);
  }

  // Add edges from links
  for (const link of workflow.links) {
    const neighbors = adjacency.get(link.source.nodeId);
    if (neighbors !== undefined) {
      neighbors.push(link.target.nodeId);
    }
  }

  return adjacency;
}

/**
 * Build a reverse adjacency list (target -> source).
 */
export function buildReverseAdjacencyList(workflow: WorkflowSchema): AdjacencyList {
  const adjacency = new Map<EntityId, EntityId[]>();

  // Initialize all nodes with empty arrays
  for (const nodeId of Object.keys(workflow.nodes)) {
    adjacency.set(nodeId, []);
  }

  // Add edges from links (reversed)
  for (const link of workflow.links) {
    const neighbors = adjacency.get(link.target.nodeId);
    if (neighbors !== undefined) {
      neighbors.push(link.source.nodeId);
    }
  }

  return adjacency;
}

// =============================================================================
// TOPOLOGICAL SORT
// =============================================================================

/**
 * Result of topological sort.
 */
export interface TopologicalSortResult {
  /** Sorted node IDs (or null if cycle detected) */
  readonly order: readonly EntityId[] | null;

  /** Whether a cycle was detected */
  readonly hasCycle: boolean;

  /** Nodes involved in cycle (if detected) */
  readonly cycleNodes?: readonly EntityId[];
}

/**
 * Perform topological sort using Kahn's algorithm.
 * Returns nodes in execution order (sources first).
 */
export function topologicalSort(workflow: WorkflowSchema): TopologicalSortResult {
  const nodeIds = Object.keys(workflow.nodes);
  const inDegree = new Map<EntityId, number>();
  const adjacency = buildAdjacencyList(workflow);

  // Calculate in-degrees
  for (const nodeId of nodeIds) {
    inDegree.set(nodeId, 0);
  }

  for (const link of workflow.links) {
    const current = inDegree.get(link.target.nodeId) ?? 0;
    inDegree.set(link.target.nodeId, current + 1);
  }

  // Find all nodes with no incoming edges
  const queue: EntityId[] = [];
  for (const [nodeId, degree] of inDegree) {
    if (degree === 0) {
      queue.push(nodeId);
    }
  }

  const order: EntityId[] = [];

  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    order.push(nodeId);

    // Reduce in-degree of neighbors
    const neighbors = adjacency.get(nodeId) ?? [];
    for (const neighbor of neighbors) {
      const newDegree = (inDegree.get(neighbor) ?? 0) - 1;
      inDegree.set(neighbor, newDegree);

      if (newDegree === 0) {
        queue.push(neighbor);
      }
    }
  }

  // Check for cycle
  if (order.length !== nodeIds.length) {
    // Find nodes still with in-degree > 0 (part of cycle)
    const cycleNodes = nodeIds.filter((id) => (inDegree.get(id) ?? 0) > 0);
    return {
      order: null,
      hasCycle: true,
      cycleNodes,
    };
  }

  return {
    order,
    hasCycle: false,
  };
}

/**
 * Perform topological sort using DFS (alternative implementation).
 * Can be used for detecting cycles with more detail.
 */
export function topologicalSortDFS(workflow: WorkflowSchema): TopologicalSortResult {
  const nodeIds = Object.keys(workflow.nodes);
  const adjacency = buildAdjacencyList(workflow);

  const WHITE = 0; // Not visited
  const GRAY = 1;  // Currently visiting (in stack)
  const BLACK = 2; // Finished

  const color = new Map<EntityId, number>();
  for (const nodeId of nodeIds) {
    color.set(nodeId, WHITE);
  }

  const result: EntityId[] = [];
  let hasCycle = false;
  const cycleNodes: EntityId[] = [];

  function visit(nodeId: EntityId): boolean {
    color.set(nodeId, GRAY);

    const neighbors = adjacency.get(nodeId) ?? [];
    for (const neighbor of neighbors) {
      const neighborColor = color.get(neighbor);

      if (neighborColor === GRAY) {
        // Found a back edge (cycle)
        hasCycle = true;
        cycleNodes.push(nodeId, neighbor);
        return false;
      }

      if (neighborColor === WHITE) {
        if (!visit(neighbor)) {
          return false;
        }
      }
    }

    color.set(nodeId, BLACK);
    result.unshift(nodeId); // Add to front (reverse post-order)
    return true;
  }

  for (const nodeId of nodeIds) {
    if (color.get(nodeId) === WHITE) {
      if (!visit(nodeId)) {
        break;
      }
    }
  }

  if (hasCycle) {
    return {
      order: null,
      hasCycle: true,
      cycleNodes,
    };
  }

  return {
    order: result,
    hasCycle: false,
  };
}

// =============================================================================
// CYCLE DETECTION
// =============================================================================

/**
 * Detect if adding a link would create a cycle.
 */
export function wouldCreateCycle(
  workflow: WorkflowSchema,
  sourceNodeId: EntityId,
  targetNodeId: EntityId
): boolean {
  // A cycle would be created if there's already a path from target to source
  return hasPath(workflow, targetNodeId, sourceNodeId);
}

/**
 * Find all cycles in the graph.
 */
export function findCycles(workflow: WorkflowSchema): readonly (readonly EntityId[])[] {
  const result = topologicalSortDFS(workflow);

  if (!result.hasCycle) {
    return [];
  }

  // Use Johnson's algorithm or DFS to find all cycles
  // For simplicity, we just return the nodes involved
  // A full implementation would return each cycle as a separate array
  if (result.cycleNodes) {
    return [result.cycleNodes];
  }
  return [];
}

// =============================================================================
// PATH FINDING
// =============================================================================

/**
 * Check if there's a path from source to target.
 */
export function hasPath(
  workflow: WorkflowSchema,
  sourceNodeId: EntityId,
  targetNodeId: EntityId
): boolean {
  if (sourceNodeId === targetNodeId) {
    return true;
  }

  const adjacency = buildAdjacencyList(workflow);
  const visited = new Set<EntityId>();
  const queue = [sourceNodeId];

  while (queue.length > 0) {
    const current = queue.shift()!;

    if (current === targetNodeId) {
      return true;
    }

    if (visited.has(current)) {
      continue;
    }
    visited.add(current);

    const neighbors = adjacency.get(current) ?? [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        queue.push(neighbor);
      }
    }
  }

  return false;
}

/**
 * Find the shortest path between two nodes.
 */
export function findPath(
  workflow: WorkflowSchema,
  sourceNodeId: EntityId,
  targetNodeId: EntityId
): readonly EntityId[] | null {
  if (sourceNodeId === targetNodeId) {
    return [sourceNodeId];
  }

  const adjacency = buildAdjacencyList(workflow);
  const visited = new Set<EntityId>();
  const parent = new Map<EntityId, EntityId>();
  const queue = [sourceNodeId];

  while (queue.length > 0) {
    const current = queue.shift()!;

    if (current === targetNodeId) {
      // Reconstruct path
      const path: EntityId[] = [current];
      let node = current;

      while (parent.has(node)) {
        node = parent.get(node)!;
        path.unshift(node);
      }

      return path;
    }

    if (visited.has(current)) {
      continue;
    }
    visited.add(current);

    const neighbors = adjacency.get(current) ?? [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        parent.set(neighbor, current);
        queue.push(neighbor);
      }
    }
  }

  return null;
}

// =============================================================================
// SUBGRAPH OPERATIONS
// =============================================================================

/**
 * Get all upstream nodes (ancestors) of a node.
 */
export function getUpstreamNodes(
  workflow: WorkflowSchema,
  nodeId: EntityId
): readonly EntityId[] {
  const reverseAdjacency = buildReverseAdjacencyList(workflow);
  const visited = new Set<EntityId>();
  const queue = [nodeId];
  const result: EntityId[] = [];

  while (queue.length > 0) {
    const current = queue.shift()!;

    if (visited.has(current)) {
      continue;
    }
    visited.add(current);

    if (current !== nodeId) {
      result.push(current);
    }

    const parents = reverseAdjacency.get(current) ?? [];
    for (const parent of parents) {
      if (!visited.has(parent)) {
        queue.push(parent);
      }
    }
  }

  return result;
}

/**
 * Get all downstream nodes (descendants) of a node.
 */
export function getDownstreamNodes(
  workflow: WorkflowSchema,
  nodeId: EntityId
): readonly EntityId[] {
  const adjacency = buildAdjacencyList(workflow);
  const visited = new Set<EntityId>();
  const queue = [nodeId];
  const result: EntityId[] = [];

  while (queue.length > 0) {
    const current = queue.shift()!;

    if (visited.has(current)) {
      continue;
    }
    visited.add(current);

    if (current !== nodeId) {
      result.push(current);
    }

    const children = adjacency.get(current) ?? [];
    for (const child of children) {
      if (!visited.has(child)) {
        queue.push(child);
      }
    }
  }

  return result;
}

/**
 * Get source nodes (nodes with no inputs).
 */
export function getSourceNodes(workflow: WorkflowSchema): readonly EntityId[] {
  const hasInput = new Set<EntityId>();

  for (const link of workflow.links) {
    hasInput.add(link.target.nodeId);
  }

  return Object.keys(workflow.nodes).filter((id) => !hasInput.has(id));
}

/**
 * Get sink nodes (nodes with no outputs connected).
 */
export function getSinkNodes(workflow: WorkflowSchema): readonly EntityId[] {
  const hasOutput = new Set<EntityId>();

  for (const link of workflow.links) {
    hasOutput.add(link.source.nodeId);
  }

  return Object.keys(workflow.nodes).filter((id) => !hasOutput.has(id));
}

// =============================================================================
// EXECUTION ORDER
// =============================================================================

/**
 * Get execution levels (nodes that can be executed in parallel).
 */
export function getExecutionLevels(workflow: WorkflowSchema): readonly EntityId[][] {
  const sortResult = topologicalSort(workflow);

  if (sortResult.order === null) {
    throw new Error('Cannot compute execution levels: graph has cycles');
  }

  const reverseAdjacency = buildReverseAdjacencyList(workflow);
  const level = new Map<EntityId, number>();

  // Compute level for each node (max level of parents + 1)
  for (const nodeId of sortResult.order) {
    const parents = reverseAdjacency.get(nodeId) ?? [];
    const parentLevels = parents.map((p) => level.get(p) ?? 0);
    const nodeLevel = parentLevels.length > 0 ? Math.max(...parentLevels) + 1 : 0;
    level.set(nodeId, nodeLevel);
  }

  // Group nodes by level
  const maxLevel = Math.max(0, ...level.values());
  const levels: EntityId[][] = Array.from({ length: maxLevel + 1 }, () => []);

  for (const [nodeId, nodeLevel] of level) {
    levels[nodeLevel]!.push(nodeId);
  }

  return levels;
}
