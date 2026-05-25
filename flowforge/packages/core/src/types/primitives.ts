/**
 * Primitive Types
 *
 * Base types and type utilities used throughout FlowForge.
 */

/**
 * Unique identifier for nodes, links, and other entities.
 * Uses string for flexibility (UUID, nanoid, or sequential).
 */
export type EntityId = string;

/**
 * Semantic version string (e.g., "1.0.0", "2.1.0-beta.1")
 */
export type SemanticVersion = `${number}.${number}.${number}${string}`;

/**
 * Supported primitive data types in the FlowForge type system.
 * These are the base types that can flow through node connections.
 */
export type PrimitiveTypeName =
  | 'ANY'       // Accepts any type (use sparingly)
  | 'BOOLEAN'   // true/false
  | 'INT'       // Integer number
  | 'FLOAT'     // Floating-point number
  | 'STRING'    // Text string
  | 'ARRAY'     // Array of values
  | 'OBJECT'    // Key-value object
  | 'BINARY'    // Binary data (Uint8Array)
  | 'NULL';     // Null/undefined value

/**
 * Type definition with optional constraints.
 * Used to define input/output types with validation.
 */
export interface TypeDefinition {
  /** Base type name */
  readonly type: PrimitiveTypeName | string;

  /** Array element type (if type is 'ARRAY') */
  readonly elementType?: TypeDefinition;

  /** Object property types (if type is 'OBJECT') */
  readonly properties?: Readonly<Record<string, TypeDefinition>>;

  /** Whether null/undefined is allowed */
  readonly nullable?: boolean;

  /** Custom type color for canvas rendering */
  readonly color?: string;
}

/**
 * Numeric constraint for INT and FLOAT types.
 */
export interface NumericConstraint {
  readonly min?: number;
  readonly max?: number;
  readonly step?: number;
  readonly precision?: number;
}

/**
 * String constraint for STRING type.
 */
export interface StringConstraint {
  readonly minLength?: number;
  readonly maxLength?: number;
  readonly pattern?: string;
  readonly format?: 'email' | 'url' | 'uuid' | 'date' | 'datetime' | 'time';
}

/**
 * Array constraint for ARRAY type.
 */
export interface ArrayConstraint {
  readonly minItems?: number;
  readonly maxItems?: number;
  readonly uniqueItems?: boolean;
}

/**
 * Union of all constraint types.
 */
export type TypeConstraint = NumericConstraint | StringConstraint | ArrayConstraint;

/**
 * Position in 2D space (for canvas placement).
 */
export interface Position {
  readonly x: number;
  readonly y: number;
}

/**
 * Size dimensions.
 */
export interface Size {
  readonly width: number;
  readonly height: number;
}

/**
 * Rectangle combining position and size.
 */
export interface Rect extends Position, Size {}

/**
 * Result type for operations that can fail.
 */
export type Result<T, E = Error> =
  | { readonly success: true; readonly value: T }
  | { readonly success: false; readonly error: E };

/**
 * Create a successful result.
 */
export function ok<T>(value: T): Result<T, never> {
  return { success: true, value };
}

/**
 * Create a failed result.
 */
export function err<E>(error: E): Result<never, E> {
  return { success: false, error };
}

/**
 * Branded type for stronger type safety.
 */
export type Brand<T, B extends string> = T & { readonly __brand: B };

/**
 * Node ID with brand for type safety.
 */
export type NodeId = Brand<EntityId, 'NodeId'>;

/**
 * Link ID with brand for type safety.
 */
export type LinkId = Brand<EntityId, 'LinkId'>;

/**
 * Group ID with brand for type safety.
 */
export type GroupId = Brand<EntityId, 'GroupId'>;

/**
 * Execution ID with brand for type safety.
 */
export type ExecutionId = Brand<EntityId, 'ExecutionId'>;

/**
 * Slot index (0-based).
 */
export type SlotIndex = number;
