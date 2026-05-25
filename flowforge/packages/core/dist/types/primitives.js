/**
 * Primitive Types
 *
 * Base types and type utilities used throughout FlowForge.
 */
/**
 * Create a successful result.
 */
export function ok(value) {
    return { success: true, value };
}
/**
 * Create a failed result.
 */
export function err(error) {
    return { success: false, error };
}
//# sourceMappingURL=primitives.js.map