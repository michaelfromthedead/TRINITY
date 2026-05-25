/**
 * Graph traversal utilities
 */

import type { LGraph } from '../src/LGraph'
import type { LGraphNode } from '../src/LGraphNode'

/**
 * Iterate over all nodes in a graph, including subgraphs
 */
export function forEachNode(
  graph: LGraph,
  callback: (node: LGraphNode, graph: LGraph) => void
): void {
  if (!graph.nodes) return

  for (const node of graph.nodes) {
    callback(node, graph)

    // Check for subgraph
    const subgraph = (node as any).subgraph
    if (subgraph) {
      forEachNode(subgraph, callback)
    }
  }
}
