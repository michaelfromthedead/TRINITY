/**
 * Workflow Schema Types
 *
 * Defines the structure of FlowForge workflows, including nodes,
 * links, groups, and metadata.
 */

import type { EntityId, SemanticVersion, Position, SlotIndex, Rect } from './primitives.js';
import type { NodeInstance } from './node.js';

// =============================================================================
// LINK DEFINITION
// =============================================================================

/**
 * Endpoint of a link (either source or target).
 */
export interface LinkEndpoint {
  /** Node instance ID */
  readonly nodeId: EntityId;

  /** Slot name (matches input/output definition name) */
  readonly slot: string;

  /** Slot index (for ordering, 0-based) */
  readonly slotIndex: SlotIndex;
}

/**
 * A connection between two nodes.
 */
export interface Link {
  /** Unique link ID */
  readonly id: EntityId;

  /** Source node and output slot */
  readonly source: LinkEndpoint;

  /** Target node and input slot */
  readonly target: LinkEndpoint;

  /** Type of data flowing through this link */
  readonly type?: string;
}

/**
 * Compact link representation for serialization.
 * [linkId, sourceNodeId, sourceSlotIndex, targetNodeId, targetSlotIndex, type?]
 */
export type CompactLink = readonly [
  id: EntityId,
  sourceNodeId: EntityId,
  sourceSlot: SlotIndex,
  targetNodeId: EntityId,
  targetSlot: SlotIndex,
  type?: string
];

// =============================================================================
// GROUP DEFINITION
// =============================================================================

/**
 * A visual group containing multiple nodes.
 */
export interface NodeGroup {
  /** Unique group ID */
  readonly id: EntityId;

  /** Display title */
  readonly title: string;

  /** Group bounds */
  readonly bounds: Rect;

  /** Group color */
  readonly color?: string;

  /** Font size for title */
  readonly fontSize?: number;

  /** Nodes contained in this group (computed or stored) */
  readonly nodeIds?: readonly EntityId[];

  /** Whether group is locked */
  readonly locked?: boolean;

  /** Custom properties */
  readonly properties?: Readonly<Record<string, unknown>>;
}

// =============================================================================
// REROUTE NODE
// =============================================================================

/**
 * A reroute node for organizing link paths.
 */
export interface RerouteNode {
  /** Unique reroute ID */
  readonly id: EntityId;

  /** Position on canvas */
  readonly position: Position;

  /** Connected link ID */
  readonly linkId: EntityId;

  /** Type of data passing through */
  readonly type?: string;
}

// =============================================================================
// WORKFLOW METADATA
// =============================================================================

/**
 * Workflow author information.
 */
export interface WorkflowAuthor {
  readonly name: string;
  readonly email?: string;
  readonly url?: string;
}

/**
 * Workflow metadata.
 */
export interface WorkflowMetadata {
  /** Workflow title */
  readonly title?: string;

  /** Workflow description */
  readonly description?: string;

  /** Workflow author(s) */
  readonly author?: WorkflowAuthor | readonly WorkflowAuthor[];

  /** Creation timestamp (ISO 8601) */
  readonly createdAt?: string;

  /** Last modified timestamp (ISO 8601) */
  readonly updatedAt?: string;

  /** Workflow version */
  readonly version?: SemanticVersion;

  /** Tags for organization */
  readonly tags?: readonly string[];

  /** License identifier */
  readonly license?: string;

  /** Thumbnail image (base64 or URL) */
  readonly thumbnail?: string;

  /** Custom metadata */
  readonly custom?: Readonly<Record<string, unknown>>;
}

// =============================================================================
// WORKFLOW CONFIGURATION
// =============================================================================

/**
 * Canvas view state.
 */
export interface CanvasViewState {
  /** Pan offset */
  readonly offset: Position;

  /** Zoom level (1.0 = 100%) */
  readonly scale: number;
}

/**
 * Workflow configuration and preferences.
 */
export interface WorkflowConfig {
  /** Canvas view state */
  readonly view?: CanvasViewState;

  /** Selected node IDs */
  readonly selectedNodes?: readonly EntityId[];

  /** Last focused node ID */
  readonly focusedNode?: EntityId;

  /** Whether grid is visible */
  readonly showGrid?: boolean;

  /** Whether links should be animated */
  readonly animateLinks?: boolean;

  /** Link curvature (0 = straight, 1 = very curved) */
  readonly linkCurvature?: number;

  /** Snap to grid settings */
  readonly snapToGrid?: {
    readonly enabled: boolean;
    readonly size: number;
  };

  /** Custom configuration */
  readonly custom?: Readonly<Record<string, unknown>>;
}

// =============================================================================
// WORKFLOW SCHEMA
// =============================================================================

/**
 * Complete workflow schema.
 * This is the root structure for serialized workflows.
 */
export interface WorkflowSchema {
  /** Schema version for migrations */
  readonly version: SemanticVersion;

  /** Workflow metadata */
  readonly metadata?: WorkflowMetadata;

  /** Workflow configuration */
  readonly config?: WorkflowConfig;

  /** Node instances indexed by ID */
  readonly nodes: Readonly<Record<EntityId, NodeInstance>>;

  /** Links between nodes */
  readonly links: readonly Link[];

  /** Visual groups */
  readonly groups?: readonly NodeGroup[];

  /** Reroute nodes */
  readonly reroutes?: readonly RerouteNode[];

  /** Required plugins for this workflow */
  readonly requiredPlugins?: readonly string[];

  /** Extra data for extensibility */
  readonly extra?: Readonly<Record<string, unknown>>;
}

/**
 * Compact workflow format for smaller file sizes.
 * Uses arrays instead of objects where possible.
 */
export interface CompactWorkflowSchema {
  /** Schema version */
  readonly v: SemanticVersion;

  /** Nodes as array */
  readonly n: readonly CompactNodeInstance[];

  /** Links as compact arrays */
  readonly l: readonly CompactLink[];

  /** Groups (optional) */
  readonly g?: readonly NodeGroup[];

  /** Metadata (optional) */
  readonly m?: WorkflowMetadata;

  /** Config (optional) */
  readonly c?: WorkflowConfig;
}

/**
 * Compact node instance for serialization.
 */
export interface CompactNodeInstance {
  /** ID */
  readonly i: EntityId;

  /** Type */
  readonly t: string;

  /** Position [x, y] */
  readonly p: readonly [number, number];

  /** Widget values */
  readonly w?: Readonly<Record<string, unknown>>;

  /** Size [width, height] */
  readonly s?: readonly [number, number];

  /** Title */
  readonly n?: string;

  /** Disabled */
  readonly d?: boolean;

  /** Collapsed */
  readonly c?: boolean;
}

// =============================================================================
// WORKFLOW UTILITIES
// =============================================================================

/**
 * Default workflow version.
 */
export const WORKFLOW_VERSION: SemanticVersion = '1.0.0';

/**
 * Create an empty workflow.
 */
export function createEmptyWorkflow(): WorkflowSchema {
  return {
    version: WORKFLOW_VERSION,
    nodes: {},
    links: [],
  };
}

/**
 * Check if a workflow is empty.
 */
export function isEmptyWorkflow(workflow: WorkflowSchema): boolean {
  return Object.keys(workflow.nodes).length === 0;
}

/**
 * Get all node IDs in a workflow.
 */
export function getNodeIds(workflow: WorkflowSchema): readonly EntityId[] {
  return Object.keys(workflow.nodes);
}

/**
 * Get all links connected to a node.
 */
export function getNodeLinks(workflow: WorkflowSchema, nodeId: EntityId): readonly Link[] {
  return workflow.links.filter(
    (link) => link.source.nodeId === nodeId || link.target.nodeId === nodeId
  );
}

/**
 * Get input links for a node.
 */
export function getInputLinks(workflow: WorkflowSchema, nodeId: EntityId): readonly Link[] {
  return workflow.links.filter((link) => link.target.nodeId === nodeId);
}

/**
 * Get output links for a node.
 */
export function getOutputLinks(workflow: WorkflowSchema, nodeId: EntityId): readonly Link[] {
  return workflow.links.filter((link) => link.source.nodeId === nodeId);
}
