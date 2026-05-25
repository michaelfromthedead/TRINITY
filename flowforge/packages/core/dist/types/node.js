/**
 * Node Definition Types
 *
 * Defines the schema for FlowForge nodes, including inputs,
 * outputs, widgets, and execution behavior.
 */
/**
 * Get ordered inputs from a node definition.
 */
export function getOrderedInputs(def) {
    return Object.entries(def.inputs);
}
/**
 * Get ordered outputs from a node definition.
 */
export function getOrderedOutputs(def) {
    return Object.entries(def.outputs);
}
//# sourceMappingURL=node.js.map