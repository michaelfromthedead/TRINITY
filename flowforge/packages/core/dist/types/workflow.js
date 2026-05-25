/**
 * Workflow Schema Types
 *
 * Defines the structure of FlowForge workflows, including nodes,
 * links, groups, and metadata.
 */
// =============================================================================
// WORKFLOW UTILITIES
// =============================================================================
/**
 * Default workflow version.
 */
export const WORKFLOW_VERSION = '1.0.0';
/**
 * Create an empty workflow.
 */
export function createEmptyWorkflow() {
    return {
        version: WORKFLOW_VERSION,
        nodes: {},
        links: [],
    };
}
/**
 * Check if a workflow is empty.
 */
export function isEmptyWorkflow(workflow) {
    return Object.keys(workflow.nodes).length === 0;
}
/**
 * Get all node IDs in a workflow.
 */
export function getNodeIds(workflow) {
    return Object.keys(workflow.nodes);
}
/**
 * Get all links connected to a node.
 */
export function getNodeLinks(workflow, nodeId) {
    return workflow.links.filter((link) => link.source.nodeId === nodeId || link.target.nodeId === nodeId);
}
/**
 * Get input links for a node.
 */
export function getInputLinks(workflow, nodeId) {
    return workflow.links.filter((link) => link.target.nodeId === nodeId);
}
/**
 * Get output links for a node.
 */
export function getOutputLinks(workflow, nodeId) {
    return workflow.links.filter((link) => link.source.nodeId === nodeId);
}
//# sourceMappingURL=workflow.js.map