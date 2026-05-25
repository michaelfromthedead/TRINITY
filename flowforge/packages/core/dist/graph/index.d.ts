/**
 * Graph Algorithms
 *
 * Algorithms for working with node graphs, including
 * topological sorting, cycle detection, and path finding.
 */
import type { EntityId } from '../types/primitives.js';
import type { WorkflowSchema } from '../types/workflow.js';
/**
 * Adjacency list representation of a graph.
 */
export type AdjacencyList = ReadonlyMap<EntityId, readonly EntityId[]>;
/**
 * Build an adjacency list from workflow links.
 * Direction: source -> target (following data flow).
 */
export declare function buildAdjacencyList(workflow: WorkflowSchema): AdjacencyList;
/**
 * Build a reverse adjacency list (target -> source).
 */
export declare function buildReverseAdjacencyList(workflow: WorkflowSchema): AdjacencyList;
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
export declare function topologicalSort(workflow: WorkflowSchema): TopologicalSortResult;
/**
 * Perform topological sort using DFS (alternative implementation).
 * Can be used for detecting cycles with more detail.
 */
export declare function topologicalSortDFS(workflow: WorkflowSchema): TopologicalSortResult;
/**
 * Detect if adding a link would create a cycle.
 */
export declare function wouldCreateCycle(workflow: WorkflowSchema, sourceNodeId: EntityId, targetNodeId: EntityId): boolean;
/**
 * Find all cycles in the graph.
 */
export declare function findCycles(workflow: WorkflowSchema): readonly (readonly EntityId[])[];
/**
 * Check if there's a path from source to target.
 */
export declare function hasPath(workflow: WorkflowSchema, sourceNodeId: EntityId, targetNodeId: EntityId): boolean;
/**
 * Find the shortest path between two nodes.
 */
export declare function findPath(workflow: WorkflowSchema, sourceNodeId: EntityId, targetNodeId: EntityId): readonly EntityId[] | null;
/**
 * Get all upstream nodes (ancestors) of a node.
 */
export declare function getUpstreamNodes(workflow: WorkflowSchema, nodeId: EntityId): readonly EntityId[];
/**
 * Get all downstream nodes (descendants) of a node.
 */
export declare function getDownstreamNodes(workflow: WorkflowSchema, nodeId: EntityId): readonly EntityId[];
/**
 * Get source nodes (nodes with no inputs).
 */
export declare function getSourceNodes(workflow: WorkflowSchema): readonly EntityId[];
/**
 * Get sink nodes (nodes with no outputs connected).
 */
export declare function getSinkNodes(workflow: WorkflowSchema): readonly EntityId[];
/**
 * Get execution levels (nodes that can be executed in parallel).
 */
export declare function getExecutionLevels(workflow: WorkflowSchema): readonly EntityId[][];
//# sourceMappingURL=index.d.ts.map