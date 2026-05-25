/**
 * Execution Context Types
 *
 * Defines types for workflow execution, including context,
 * progress tracking, and result handling.
 */
import type { EntityId, ExecutionId } from './primitives.js';
import type { WorkflowSchema } from './workflow.js';
import type { NodeDefinition, WidgetValues } from './node.js';
/**
 * Execution priority levels.
 */
export type ExecutionPriority = 'low' | 'normal' | 'high' | 'critical';
/**
 * Execution mode determines how nodes are processed.
 */
export type ExecutionMode = 'full' | 'partial' | 'single';
/**
 * Configuration for workflow execution.
 */
export interface ExecutionConfig {
    /** Execution mode */
    readonly mode: ExecutionMode;
    /** Priority level */
    readonly priority?: ExecutionPriority;
    /** Timeout in milliseconds (0 = no timeout) */
    readonly timeout?: number;
    /** Whether to cache intermediate results */
    readonly enableCaching?: boolean;
    /** Maximum concurrent node executions */
    readonly maxConcurrency?: number;
    /** Specific nodes to execute (for partial/single mode) */
    readonly targetNodes?: readonly EntityId[];
    /** Whether to validate types before execution */
    readonly validateTypes?: boolean;
    /** Whether to collect execution metrics */
    readonly collectMetrics?: boolean;
    /** Custom execution options */
    readonly options?: Readonly<Record<string, unknown>>;
}
/**
 * Logger interface for execution context.
 */
export interface ExecutionLogger {
    debug(message: string, data?: unknown): void;
    info(message: string, data?: unknown): void;
    warn(message: string, data?: unknown): void;
    error(message: string, error?: Error | unknown): void;
}
/**
 * Execution context passed to each node during execution.
 * This provides access to workflow state, outputs, and utilities.
 */
export interface ExecutionContext {
    /** Unique execution ID */
    readonly executionId: ExecutionId;
    /** Workflow being executed */
    readonly workflow: WorkflowSchema;
    /** Execution configuration */
    readonly config: ExecutionConfig;
    /** Current node being executed */
    readonly currentNodeId: EntityId;
    /** Get output from a previously executed node */
    getNodeOutput<T = unknown>(nodeId: EntityId, outputName: string): T | undefined;
    /** Check if a node has been executed */
    isNodeExecuted(nodeId: EntityId): boolean;
    /** Get all outputs from a node */
    getAllNodeOutputs(nodeId: EntityId): Readonly<Record<string, unknown>> | undefined;
    /** Report execution progress (0-100) */
    reportProgress(progress: number, message?: string): void;
    /** Request cancellation check */
    isCancelled(): boolean;
    /** Request a pause check */
    isPaused(): boolean;
    /** Wait if execution is paused */
    waitIfPaused(): Promise<void>;
    /** Logger for execution messages */
    readonly logger: ExecutionLogger;
    /** Access to node definitions */
    getNodeDefinition(type: string): NodeDefinition | undefined;
    /** Store temporary data for this execution */
    setExecutionData<T>(key: string, value: T): void;
    /** Retrieve temporary execution data */
    getExecutionData<T>(key: string): T | undefined;
    /** Emit a custom event during execution */
    emit(event: string, payload?: unknown): void;
}
/**
 * Input values collected from connected nodes.
 */
export type NodeInputValues = Readonly<Record<string, unknown>>;
/**
 * Output values produced by a node.
 */
export type NodeOutputValues = Readonly<Record<string, unknown>>;
/**
 * Node execution function signature.
 */
export type NodeExecuteFunction = (inputs: NodeInputValues, widgets: WidgetValues, context: ExecutionContext) => NodeOutputValues | Promise<NodeOutputValues>;
/**
 * Node validation function signature.
 */
export type NodeValidateFunction = (inputs: NodeInputValues, widgets: WidgetValues) => ValidationResult;
/**
 * Validation result from a node.
 */
export interface ValidationResult {
    readonly valid: boolean;
    readonly errors?: readonly ValidationError[];
    readonly warnings?: readonly ValidationWarning[];
}
/**
 * Validation error.
 */
export interface ValidationError {
    readonly field: string;
    readonly message: string;
    readonly code?: string;
}
/**
 * Validation warning.
 */
export interface ValidationWarning {
    readonly field: string;
    readonly message: string;
    readonly code?: string;
}
/**
 * Execution event types.
 */
export type ExecutionEventType = 'execution:start' | 'execution:progress' | 'execution:complete' | 'execution:error' | 'execution:cancelled' | 'node:start' | 'node:progress' | 'node:complete' | 'node:error' | 'node:skipped';
/**
 * Base execution event.
 */
interface BaseExecutionEvent {
    readonly type: ExecutionEventType;
    readonly executionId: ExecutionId;
    readonly timestamp: number;
}
/**
 * Workflow execution started.
 */
export interface ExecutionStartEvent extends BaseExecutionEvent {
    readonly type: 'execution:start';
    readonly totalNodes: number;
    readonly config: ExecutionConfig;
}
/**
 * Workflow execution progress.
 */
export interface ExecutionProgressEvent extends BaseExecutionEvent {
    readonly type: 'execution:progress';
    readonly completedNodes: number;
    readonly totalNodes: number;
    readonly percentage: number;
}
/**
 * Workflow execution completed.
 */
export interface ExecutionCompleteEvent extends BaseExecutionEvent {
    readonly type: 'execution:complete';
    readonly outputs: Readonly<Record<EntityId, NodeOutputValues>>;
    readonly duration: number;
    readonly metrics?: ExecutionMetrics;
}
/**
 * Workflow execution error.
 */
export interface ExecutionErrorEvent extends BaseExecutionEvent {
    readonly type: 'execution:error';
    readonly error: ExecutionError;
    readonly nodeId?: EntityId;
}
/**
 * Workflow execution cancelled.
 */
export interface ExecutionCancelledEvent extends BaseExecutionEvent {
    readonly type: 'execution:cancelled';
    readonly reason?: string;
    readonly completedNodes: number;
}
/**
 * Node execution started.
 */
export interface NodeStartEvent extends BaseExecutionEvent {
    readonly type: 'node:start';
    readonly nodeId: EntityId;
    readonly nodeType: string;
}
/**
 * Node execution progress.
 */
export interface NodeProgressEvent extends BaseExecutionEvent {
    readonly type: 'node:progress';
    readonly nodeId: EntityId;
    readonly progress: number;
    readonly message?: string;
}
/**
 * Node execution completed.
 */
export interface NodeCompleteEvent extends BaseExecutionEvent {
    readonly type: 'node:complete';
    readonly nodeId: EntityId;
    readonly outputs: NodeOutputValues;
    readonly duration: number;
}
/**
 * Node execution error.
 */
export interface NodeErrorEvent extends BaseExecutionEvent {
    readonly type: 'node:error';
    readonly nodeId: EntityId;
    readonly error: ExecutionError;
}
/**
 * Node execution skipped.
 */
export interface NodeSkippedEvent extends BaseExecutionEvent {
    readonly type: 'node:skipped';
    readonly nodeId: EntityId;
    readonly reason: 'disabled' | 'bypass' | 'conditional' | 'cached';
}
/**
 * Union of all execution events.
 */
export type ExecutionEvent = ExecutionStartEvent | ExecutionProgressEvent | ExecutionCompleteEvent | ExecutionErrorEvent | ExecutionCancelledEvent | NodeStartEvent | NodeProgressEvent | NodeCompleteEvent | NodeErrorEvent | NodeSkippedEvent;
/**
 * Execution error details.
 */
export interface ExecutionError {
    readonly code: string;
    readonly message: string;
    readonly nodeId?: EntityId;
    readonly inputName?: string;
    readonly stack?: string;
    readonly cause?: ExecutionError;
}
/**
 * Execution metrics for performance analysis.
 */
export interface ExecutionMetrics {
    /** Total execution duration in ms */
    readonly totalDuration: number;
    /** Per-node execution times in ms */
    readonly nodeDurations: Readonly<Record<EntityId, number>>;
    /** Peak memory usage in bytes */
    readonly peakMemory?: number;
    /** Number of cache hits */
    readonly cacheHits?: number;
    /** Number of cache misses */
    readonly cacheMisses?: number;
    /** Custom metrics */
    readonly custom?: Readonly<Record<string, number>>;
}
/**
 * Result of a workflow execution.
 */
export interface ExecutionResult {
    /** Execution ID */
    readonly executionId: ExecutionId;
    /** Whether execution succeeded */
    readonly success: boolean;
    /** All node outputs (if successful) */
    readonly outputs?: Readonly<Record<EntityId, NodeOutputValues>>;
    /** Error details (if failed) */
    readonly error?: ExecutionError;
    /** Execution metrics */
    readonly metrics?: ExecutionMetrics;
    /** Execution duration in ms */
    readonly duration: number;
    /** Timestamp when execution completed */
    readonly completedAt: string;
}
/**
 * Queued execution item.
 */
export interface QueuedExecution {
    readonly executionId: ExecutionId;
    readonly workflow: WorkflowSchema;
    readonly config: ExecutionConfig;
    readonly queuedAt: string;
    readonly priority: ExecutionPriority;
}
/**
 * Execution queue status.
 */
export interface ExecutionQueueStatus {
    /** Currently running execution (if any) */
    readonly running?: {
        readonly executionId: ExecutionId;
        readonly startedAt: string;
        readonly progress: number;
    };
    /** Queued executions waiting to run */
    readonly pending: readonly QueuedExecution[];
    /** Recently completed executions */
    readonly recent: readonly {
        readonly executionId: ExecutionId;
        readonly success: boolean;
        readonly completedAt: string;
        readonly duration: number;
    }[];
}
export {};
//# sourceMappingURL=execution.d.ts.map