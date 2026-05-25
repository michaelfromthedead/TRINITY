/**
 * Node Definition Types
 *
 * Defines the schema for FlowForge nodes, including inputs,
 * outputs, widgets, and execution behavior.
 */
import type { EntityId, TypeDefinition, TypeConstraint, Position, Size } from './primitives.js';
/**
 * Defines a node input slot.
 * Inputs receive data from connected output slots.
 */
export interface InputDefinition {
    /** Display name for the input */
    readonly name: string;
    /** Type of data this input accepts */
    readonly type: TypeDefinition | string;
    /** Whether this input is required for execution */
    readonly required?: boolean;
    /** Default value if not connected and not required */
    readonly defaultValue?: unknown;
    /** Human-readable description */
    readonly description?: string;
    /** Allow multiple connections (creates array of values) */
    readonly multi?: boolean;
    /** Type constraints (min, max, pattern, etc.) */
    readonly constraints?: TypeConstraint;
    /** Hidden from UI but still functional */
    readonly hidden?: boolean;
}
/**
 * Defines a node output slot.
 * Outputs send data to connected input slots.
 */
export interface OutputDefinition {
    /** Display name for the output */
    readonly name: string;
    /** Type of data this output produces */
    readonly type: TypeDefinition | string;
    /** Human-readable description */
    readonly description?: string;
    /** Whether this output is the "primary" output (for preview) */
    readonly primary?: boolean;
    /** Hidden from UI but still functional */
    readonly hidden?: boolean;
}
/**
 * Base widget definition shared by all widget types.
 */
interface BaseWidgetDefinition {
    /** Unique name for this widget (used to get/set values) */
    readonly name: string;
    /** Display label */
    readonly label?: string;
    /** Human-readable description */
    readonly description?: string;
    /** Default value */
    readonly defaultValue?: unknown;
    /** Whether this widget is disabled */
    readonly disabled?: boolean;
    /** Whether this widget is hidden */
    readonly hidden?: boolean;
}
/**
 * Number input widget (slider or field).
 */
export interface NumberWidget extends BaseWidgetDefinition {
    readonly type: 'number';
    readonly min?: number;
    readonly max?: number;
    readonly step?: number;
    readonly precision?: number;
    readonly displayMode?: 'slider' | 'field' | 'both';
    readonly defaultValue?: number;
}
/**
 * Integer input widget.
 */
export interface IntegerWidget extends BaseWidgetDefinition {
    readonly type: 'integer';
    readonly min?: number;
    readonly max?: number;
    readonly step?: number;
    readonly displayMode?: 'slider' | 'field' | 'both';
    readonly defaultValue?: number;
}
/**
 * Text input widget.
 */
export interface TextWidget extends BaseWidgetDefinition {
    readonly type: 'text';
    readonly placeholder?: string;
    readonly multiline?: boolean;
    readonly rows?: number;
    readonly maxLength?: number;
    readonly defaultValue?: string;
}
/**
 * Boolean toggle widget.
 */
export interface BooleanWidget extends BaseWidgetDefinition {
    readonly type: 'boolean';
    readonly defaultValue?: boolean;
}
/**
 * Dropdown selection widget.
 */
export interface SelectWidget extends BaseWidgetDefinition {
    readonly type: 'select';
    readonly options: readonly (string | {
        value: string;
        label: string;
    })[];
    readonly multiple?: boolean;
    readonly defaultValue?: string | readonly string[];
}
/**
 * Color picker widget.
 */
export interface ColorWidget extends BaseWidgetDefinition {
    readonly type: 'color';
    readonly format?: 'hex' | 'rgb' | 'rgba' | 'hsl';
    readonly defaultValue?: string;
}
/**
 * File path widget.
 */
export interface FileWidget extends BaseWidgetDefinition {
    readonly type: 'file';
    readonly accept?: readonly string[];
    readonly directory?: boolean;
    readonly multiple?: boolean;
    readonly defaultValue?: string | readonly string[];
}
/**
 * Custom widget for plugin-defined UI.
 */
export interface CustomWidget extends BaseWidgetDefinition {
    readonly type: 'custom';
    readonly component: string;
    readonly props?: Readonly<Record<string, unknown>>;
}
/**
 * Union of all widget types.
 */
export type WidgetDefinition = NumberWidget | IntegerWidget | TextWidget | BooleanWidget | SelectWidget | ColorWidget | FileWidget | CustomWidget;
/**
 * Node category for organization in the node browser.
 */
export interface NodeCategory {
    /** Category path (e.g., "Math/Arithmetic") */
    readonly path: string;
    /** Optional icon */
    readonly icon?: string;
    /** Category description */
    readonly description?: string;
}
/**
 * Complete node definition.
 * This is the schema that defines a node type.
 */
export interface NodeDefinition {
    /** Unique type identifier (e.g., "Math/Add", "Logic/If") */
    readonly type: string;
    /** Display name (defaults to last segment of type) */
    readonly displayName?: string;
    /** Category for node browser organization */
    readonly category: string | NodeCategory;
    /** Human-readable description */
    readonly description?: string;
    /** Documentation URL */
    readonly docsUrl?: string;
    /** Input slot definitions */
    readonly inputs: Readonly<Record<string, InputDefinition>>;
    /** Output slot definitions */
    readonly outputs: Readonly<Record<string, OutputDefinition>>;
    /** Widget definitions for node configuration */
    readonly widgets?: Readonly<Record<string, WidgetDefinition>>;
    /** Node color (hex or CSS color name) */
    readonly color?: string;
    /** Background color */
    readonly bgColor?: string;
    /** Default size */
    readonly size?: Size;
    /** Whether this node is deprecated */
    readonly deprecated?: boolean;
    /** Replacement node type if deprecated */
    readonly replacedBy?: string;
    /** Node execution mode */
    readonly executionMode?: 'sync' | 'async' | 'stream';
    /** Tags for search and filtering */
    readonly tags?: readonly string[];
    /** Plugin that provides this node */
    readonly providedBy?: string;
    /** Whether this node can be cached */
    readonly cacheable?: boolean;
    /** Estimated execution cost (for optimization) */
    readonly executionCost?: 'low' | 'medium' | 'high' | 'very-high';
}
/**
 * Widget values for a node instance.
 */
export type WidgetValues = Readonly<Record<string, unknown>>;
/**
 * Instance of a node in a workflow.
 * This represents a specific node placed on the canvas.
 */
export interface NodeInstance {
    /** Unique instance ID */
    readonly id: EntityId;
    /** Node type (references NodeDefinition.type) */
    readonly type: string;
    /** Position on canvas */
    readonly position: Position;
    /** Optional custom size override */
    readonly size?: Size;
    /** Widget values */
    readonly widgets: WidgetValues;
    /** Optional display title override */
    readonly title?: string;
    /** Execution order hint (optional, computed if not set) */
    readonly order?: number;
    /** Whether this node is currently disabled */
    readonly disabled?: boolean;
    /** Whether this node is collapsed */
    readonly collapsed?: boolean;
    /** Custom node-level properties */
    readonly properties?: Readonly<Record<string, unknown>>;
    /** Visual mode (default, bypass, mute) */
    readonly mode?: 'default' | 'bypass' | 'mute';
}
/**
 * Ordered inputs for a node definition (for UI rendering).
 */
export type OrderedInputs = readonly (readonly [string, InputDefinition])[];
/**
 * Ordered outputs for a node definition (for UI rendering).
 */
export type OrderedOutputs = readonly (readonly [string, OutputDefinition])[];
/**
 * Get ordered inputs from a node definition.
 */
export declare function getOrderedInputs(def: NodeDefinition): OrderedInputs;
/**
 * Get ordered outputs from a node definition.
 */
export declare function getOrderedOutputs(def: NodeDefinition): OrderedOutputs;
/**
 * Type for a dictionary of node definitions indexed by type.
 */
export type NodeDefinitionMap = Readonly<Record<string, NodeDefinition>>;
export {};
//# sourceMappingURL=node.d.ts.map