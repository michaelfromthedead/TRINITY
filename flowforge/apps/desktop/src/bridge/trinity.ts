/**
 * Trinity Introspection Bridge
 *
 * Provides TypeScript functions for communicating with the Trinity runtime
 * to query ECS state, registry contents, and runtime events.
 */

import { invoke } from '@tauri-apps/api/core';
import { isTauri } from '@/services';

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Trinity runtime status information.
 */
export interface TrinityStatus {
  /** Whether Trinity runtime is available and responding */
  available: boolean;
  /** Trinity runtime version string */
  version: string | null;
  /** Number of registered types in the registry */
  registeredTypes: number;
  /** Number of active entity instances */
  activeInstances: number;
  /** Timestamp of the status check */
  timestamp: number;
}

/**
 * Entry types in the Trinity registry.
 */
export type RegistryEntryType = 'component' | 'system' | 'resource' | 'event';

/**
 * A single entry in the Trinity type registry.
 */
export interface RegistryEntry {
  /** Unique identifier for this registry entry */
  id: string;
  /** Name of the registered type */
  name: string;
  /** Type category (component, system, resource, event) */
  type: RegistryEntryType;
  /** Python module path where the type is defined */
  module: string;
  /** Optional description from docstring */
  description?: string;
  /** List of field names for components */
  fields?: string[];
  /** List of method names for systems */
  methods?: string[];
  /** Timestamp when the type was registered */
  registeredAt: number;
}

/**
 * Complete registry contents response.
 */
export interface RegistryContents {
  /** All registered entries */
  entries: RegistryEntry[];
  /** Total count of entries */
  totalCount: number;
  /** Counts by entry type */
  countByType: Record<RegistryEntryType, number>;
  /** Timestamp when the registry was queried */
  queriedAt: number;
}

/**
 * An active instance in the Trinity runtime.
 */
export interface TrinityInstance {
  /** Unique instance identifier (entity ID for components) */
  id: string;
  /** Name of the component/type this instance belongs to */
  componentName: string;
  /** Type category */
  type: RegistryEntryType;
  /** Current field values (for components) */
  data: Record<string, unknown>;
  /** Entity ID this instance is attached to (for components) */
  entityId?: number;
  /** Timestamp when the instance was created */
  createdAt: number;
  /** Timestamp of last update */
  updatedAt: number;
}

/**
 * Query response for active instances.
 */
export interface InstancesQueryResult {
  /** Matching instances */
  instances: TrinityInstance[];
  /** Total count of matching instances */
  totalCount: number;
  /** Filter applied (if any) */
  filter?: string;
  /** Timestamp of the query */
  queriedAt: number;
}

/**
 * Event severity levels.
 */
export type EventSeverity = 'debug' | 'info' | 'warning' | 'error';

/**
 * A Trinity runtime event.
 */
export interface TrinityEvent {
  /** Unique event identifier */
  id: string;
  /** Event type name */
  eventType: string;
  /** Human-readable message */
  message: string;
  /** Event severity level */
  severity: EventSeverity;
  /** Event payload data */
  payload: Record<string, unknown>;
  /** Source component/system that emitted the event */
  source?: string;
  /** Timestamp when the event occurred */
  timestamp: number;
}

/**
 * Response containing recent events.
 */
export interface RecentEventsResult {
  /** List of recent events */
  events: TrinityEvent[];
  /** Total count of events returned */
  count: number;
  /** Whether there are more events available */
  hasMore: boolean;
  /** Timestamp of the query */
  queriedAt: number;
}

// =============================================================================
// Bridge Functions
// =============================================================================

/**
 * Connection result from Trinity connect operation.
 */
export interface TrinityConnectionResult {
  /** Whether the connection was successful */
  success: boolean;
  /** Error message if connection failed */
  error?: string;
  /** Session ID if connection succeeded */
  sessionId?: string;
}

/**
 * Connect to the Trinity runtime.
 * This should be called before checking status to establish the connection.
 *
 * @returns Connection result with success status and optional error/sessionId
 */
export async function connectTrinity(): Promise<TrinityConnectionResult> {
  if (!isTauri()) {
    console.warn('[Trinity] Not available in browser mode');
    return { success: false, error: 'Trinity requires Tauri desktop environment' };
  }

  try {
    const result = await invoke<TrinityConnectionResult>('trinity_connect');
    return result;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('[Trinity] Connection failed:', errorMessage);
    return {
      success: false,
      error: errorMessage,
    };
  }
}

/**
 * Check if Trinity runtime is available and get status information.
 * This is a lightweight call suitable for polling.
 *
 * @returns Trinity status information including availability, version, counts, and optional error
 */
export async function checkTrinityStatus(): Promise<TrinityStatus & { error?: string }> {
  if (!isTauri()) {
    return {
      available: false,
      version: null,
      registeredTypes: 0,
      activeInstances: 0,
      timestamp: Date.now(),
      error: 'Trinity requires Tauri desktop environment',
    };
  }

  try {
    const result = await invoke<TrinityStatus>('trinity_status');
    return result;
  } catch (error) {
    // If the backend command doesn't exist or fails, Trinity is not available
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.warn('[Trinity] Status check failed:', errorMessage);
    return {
      available: false,
      version: null,
      registeredTypes: 0,
      activeInstances: 0,
      timestamp: Date.now(),
      error: errorMessage,
    };
  }
}

/**
 * Fetch all registered types from the Trinity registry.
 * Returns components, systems, resources, and events.
 *
 * @returns Complete registry contents with type information
 * @throws Error if Trinity is not available or the query fails
 */
export async function getRegistryContents(): Promise<RegistryContents> {
  try {
    const result = await invoke<RegistryContents>('trinity_registry_list');
    return result;
  } catch (error) {
    console.error('[Trinity] Failed to get registry contents:', error);
    throw new Error(
      `Failed to fetch Trinity registry: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Query active instances in the Trinity runtime.
 * Optionally filter by component name.
 *
 * @param componentName - Optional component name to filter by
 * @returns Query result containing matching instances
 * @throws Error if Trinity is not available or the query fails
 */
export async function queryInstances(
  componentName?: string
): Promise<InstancesQueryResult> {
  try {
    const result = await invoke<InstancesQueryResult>('trinity_instances_query', {
      request: { componentName },
    });
    return result;
  } catch (error) {
    console.error('[Trinity] Failed to query instances:', error);
    throw new Error(
      `Failed to query Trinity instances: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Get recent events from the Trinity runtime.
 *
 * @param limit - Maximum number of events to return (default: 50)
 * @returns Recent events result
 * @throws Error if Trinity is not available or the query fails
 */
export async function getRecentEvents(limit?: number): Promise<RecentEventsResult> {
  try {
    const result = await invoke<RecentEventsResult>('trinity_events_recent', {
      request: { limit: limit ?? 50 },
    });
    return result;
  } catch (error) {
    console.error('[Trinity] Failed to get recent events:', error);
    throw new Error(
      `Failed to fetch Trinity events: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Filter registry entries by type.
 */
export function filterEntriesByType(
  entries: RegistryEntry[],
  type: RegistryEntryType
): RegistryEntry[] {
  return entries.filter((entry) => entry.type === type);
}

/**
 * Get a summary of registry counts by type.
 */
export function getRegistrySummary(contents: RegistryContents): {
  components: number;
  systems: number;
  resources: number;
  events: number;
  total: number;
} {
  return {
    components: contents.countByType.component ?? 0,
    systems: contents.countByType.system ?? 0,
    resources: contents.countByType.resource ?? 0,
    events: contents.countByType.event ?? 0,
    total: contents.totalCount,
  };
}

// =============================================================================
// Inspection Types
// =============================================================================

/**
 * Detailed inspection result for a Trinity type.
 */
export interface InspectionResult {
  /** Whether the inspection was successful */
  success: boolean;
  /** Error message if inspection failed */
  error?: string;
  /** The type name */
  name?: string;
  /** Fully qualified name (module.ClassName) */
  qualifiedName?: string;
  /** Type category (component, system, resource, event, etc.) */
  category?: string;
  /** Module where the type is defined */
  module?: string;
  /** Documentation string */
  doc?: string;
  /** Source file information */
  source?: {
    file: string;
    line: number | null;
  };
  /** Metaclass name */
  metaclass?: string;
  /** Component hierarchy (base classes) */
  hierarchy?: HierarchyEntry[];
  /** Decorator chain */
  decorators?: DecoratorEntry[];
  /** Field types for components */
  fieldTypes?: Record<string, string>;
  /** Field defaults */
  fieldDefaults?: Record<string, unknown>;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Entry in the class hierarchy.
 */
export interface HierarchyEntry {
  /** Class name */
  name: string;
  /** Module path */
  module?: string;
  /** Whether this is a Trinity base class */
  isTrinityBase: boolean;
}

/**
 * Entry in the decorator chain.
 */
export interface DecoratorEntry {
  /** Decorator name */
  name: string;
  /** Tier level (1-5) */
  tier?: number;
  /** Tier name */
  tierName?: string;
  /** Whether it's a foundation decorator */
  foundation?: boolean;
  /** Documentation */
  doc?: string;
  /** Arguments passed to the decorator */
  args?: Record<string, unknown>;
}

// =============================================================================
// Inspection Functions
// =============================================================================

/**
 * Inspect a Trinity type by name.
 * Returns detailed information including hierarchy, decorators, metaclass, etc.
 *
 * @param typeName - The type name to inspect (can be simple or qualified name)
 * @returns Detailed inspection result
 * @throws Error if the backend is not available
 */
export async function inspectType(typeName: string): Promise<InspectionResult> {
  try {
    const result = await invoke<InspectionResult>('trinity_inspect', {
      request: { typeName },
    });
    return result;
  } catch (error) {
    console.error('[Trinity] Failed to inspect type:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Get detailed inspector information for a target.
 * Unified API that can inspect types, instances, or decorators.
 *
 * @param targetType - Type of target ("type", "instance", "decorator")
 * @param options - Additional options based on target type
 * @returns Inspection result
 */
export async function inspectorGet(
  targetType: 'type' | 'instance' | 'decorator',
  options: {
    qualifiedName?: string;
    targetId?: number;
  }
): Promise<InspectionResult> {
  try {
    const result = await invoke<InspectionResult>('trinity_inspector_get', {
      request: {
        targetType,
        qualifiedName: options.qualifiedName,
        targetId: options.targetId,
      },
    });
    return result;
  } catch (error) {
    console.error('[Trinity] Failed to get inspector data:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
