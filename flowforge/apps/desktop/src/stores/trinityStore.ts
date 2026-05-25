/**
 * Trinity Store - Pinia store for Trinity runtime introspection
 *
 * Manages Trinity runtime state including registry contents, active instances,
 * and runtime events with automatic polling support.
 */
import { defineStore } from 'pinia';
import { computed, ref, shallowRef } from 'vue';
import { TRINITY_CONFIG } from '@/config/flowforge.config';
import {
  checkTrinityStatus,
  connectTrinity,
  getRegistryContents,
  queryInstances,
  getRecentEvents,
  type TrinityStatus,
  type TrinityConnectionResult,
  type RegistryEntry,
  type RegistryEntryType,
  type TrinityInstance,
  type TrinityEvent,
  type RegistryContents,
  type InstancesQueryResult,
  type RecentEventsResult,
} from '@/bridge/trinity';

// =============================================================================
// Event Bus for Trinity Events
// =============================================================================

/**
 * Custom event for when new Trinity events are detected.
 */
export const TRINITY_NEW_EVENTS = 'trinity:new-events';

/**
 * Detail type for the new events custom event.
 */
export interface TrinityNewEventsDetail {
  /** Newly detected events */
  events: TrinityEvent[];
  /** Timestamp when the events were detected */
  timestamp: number;
}

/**
 * Dispatch a custom event when new Trinity events are detected.
 */
function dispatchNewEventsEvent(newEvents: TrinityEvent[]): void {
  if (newEvents.length === 0) return;

  const detail: TrinityNewEventsDetail = {
    events: newEvents,
    timestamp: Date.now(),
  };

  window.dispatchEvent(new CustomEvent(TRINITY_NEW_EVENTS, { detail }));
}

export const useTrinityStore = defineStore('trinity', () => {
  // ===========================================================================
  // State
  // ===========================================================================

  /**
   * Whether Trinity runtime is available.
   */
  const isAvailable = ref(false);

  /**
   * Trinity runtime version string.
   */
  const version = ref<string | null>(null);

  /**
   * Registry entries from Trinity.
   * Using shallowRef for performance with large arrays.
   */
  const registryEntries = shallowRef<RegistryEntry[]>([]);

  /**
   * Active instances in Trinity runtime.
   */
  const instances = shallowRef<TrinityInstance[]>([]);

  /**
   * Recent events from Trinity runtime.
   */
  const events = shallowRef<TrinityEvent[]>([]);

  /**
   * Timestamp of the last successful update.
   */
  const lastUpdated = ref<number | null>(null);

  /**
   * Loading states for different operations.
   */
  const loading = ref({
    status: false,
    registry: false,
    instances: false,
    events: false,
  });

  /**
   * Error messages for different operations.
   */
  const errors = ref({
    status: null as string | null,
    registry: null as string | null,
    instances: null as string | null,
    events: null as string | null,
  });

  /**
   * Connection error message (set when initial connection fails).
   * This provides a meaningful error to display to the user.
   */
  const connectionError = ref<string | null>(null);

  /**
   * Polling interval ID (null when not polling).
   */
  let pollingIntervalId: ReturnType<typeof setInterval> | null = null;

  /**
   * Whether polling is currently active.
   */
  const isPolling = ref(false);

  // ===========================================================================
  // Computed Properties
  // ===========================================================================

  /**
   * Count of registered components.
   */
  const componentCount = computed(() => {
    return registryEntries.value.filter((e) => e.type === 'component').length;
  });

  /**
   * Count of registered systems.
   */
  const systemCount = computed(() => {
    return registryEntries.value.filter((e) => e.type === 'system').length;
  });

  /**
   * Count of registered resources.
   */
  const resourceCount = computed(() => {
    return registryEntries.value.filter((e) => e.type === 'resource').length;
  });

  /**
   * Count of recent events.
   */
  const eventCount = computed(() => {
    return events.value.length;
  });

  /**
   * Total count of all registry entries.
   */
  const totalRegistryCount = computed(() => {
    return registryEntries.value.length;
  });

  /**
   * Total count of active instances.
   */
  const totalInstanceCount = computed(() => {
    return instances.value.length;
  });

  /**
   * Whether any loading operation is in progress.
   */
  const isLoading = computed(() => {
    return (
      loading.value.status ||
      loading.value.registry ||
      loading.value.instances ||
      loading.value.events
    );
  });

  /**
   * Whether there are any errors.
   */
  const hasErrors = computed(() => {
    return (
      errors.value.status !== null ||
      errors.value.registry !== null ||
      errors.value.instances !== null ||
      errors.value.events !== null
    );
  });

  /**
   * Get registry entries filtered by type.
   */
  const getEntriesByType = computed(() => {
    return (type: RegistryEntryType): RegistryEntry[] => {
      return registryEntries.value.filter((e) => e.type === type);
    };
  });

  /**
   * Get instances filtered by component name.
   */
  const getInstancesByComponent = computed(() => {
    return (componentName: string): TrinityInstance[] => {
      return instances.value.filter((i) => i.componentName === componentName);
    };
  });

  // ===========================================================================
  // Actions
  // ===========================================================================

  /**
   * Check Trinity status and update availability.
   */
  async function checkStatus(): Promise<TrinityStatus> {
    loading.value.status = true;
    errors.value.status = null;

    try {
      const status = await checkTrinityStatus();
      isAvailable.value = status.available;
      version.value = status.version;
      lastUpdated.value = Date.now();
      return status;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      errors.value.status = message;
      isAvailable.value = false;
      console.error('[TrinityStore] Status check failed:', error);
      throw error;
    } finally {
      loading.value.status = false;
    }
  }

  /**
   * Refresh registry contents from Trinity.
   */
  async function refreshRegistry(): Promise<RegistryContents> {
    loading.value.registry = true;
    errors.value.registry = null;

    try {
      const contents = await getRegistryContents();
      registryEntries.value = contents.entries;
      lastUpdated.value = Date.now();
      return contents;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      errors.value.registry = message;
      console.error('[TrinityStore] Registry refresh failed:', error);
      throw error;
    } finally {
      loading.value.registry = false;
    }
  }

  /**
   * Refresh active instances from Trinity.
   * @param componentName - Optional component name filter
   */
  async function refreshInstances(
    componentName?: string
  ): Promise<InstancesQueryResult> {
    loading.value.instances = true;
    errors.value.instances = null;

    try {
      const result = await queryInstances(componentName);
      instances.value = result.instances;
      lastUpdated.value = Date.now();
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      errors.value.instances = message;
      console.error('[TrinityStore] Instances refresh failed:', error);
      throw error;
    } finally {
      loading.value.instances = false;
    }
  }

  /**
   * Refresh recent events from Trinity.
   * Detects new events and emits a custom event for highlighting.
   * @param limit - Maximum number of events to fetch
   */
  async function refreshEvents(limit?: number): Promise<RecentEventsResult> {
    loading.value.events = true;
    errors.value.events = null;

    const maxEvents = limit ?? TRINITY_CONFIG.maxEvents;

    try {
      const result = await getRecentEvents(maxEvents);

      // Detect new events by comparing IDs
      const previousEventIds = new Set(events.value.map((e) => e.id));
      const newEvents = result.events.filter((e) => !previousEventIds.has(e.id));

      // Update events state
      events.value = result.events;
      lastUpdated.value = Date.now();

      // Dispatch event for new events (for highlighting)
      if (newEvents.length > 0) {
        dispatchNewEventsEvent(newEvents);
      }

      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      errors.value.events = message;
      console.error('[TrinityStore] Events refresh failed:', error);
      throw error;
    } finally {
      loading.value.events = false;
    }
  }

  /**
   * Refresh all Trinity data (status, registry, instances, events).
   */
  async function refreshAll(): Promise<void> {
    // Check status first
    await checkStatus();

    // Only fetch other data if Trinity is available
    if (isAvailable.value) {
      await Promise.all([
        refreshRegistry().catch((e) =>
          console.warn('[TrinityStore] Registry refresh failed:', e)
        ),
        refreshInstances().catch((e) =>
          console.warn('[TrinityStore] Instances refresh failed:', e)
        ),
        refreshEvents().catch((e) =>
          console.warn('[TrinityStore] Events refresh failed:', e)
        ),
      ]);
    }
  }

  /**
   * Start polling for Trinity updates.
   * @param interval - Polling interval in milliseconds (default from config)
   */
  function startPolling(interval?: number): void {
    if (isPolling.value) {
      console.warn('[TrinityStore] Polling is already active');
      return;
    }

    const pollInterval = interval ?? TRINITY_CONFIG.pollingInterval;

    // Do an initial refresh
    refreshAll().catch((e) =>
      console.warn('[TrinityStore] Initial refresh failed:', e)
    );

    // Set up the polling interval
    pollingIntervalId = setInterval(() => {
      refreshAll().catch((e) =>
        console.warn('[TrinityStore] Polling refresh failed:', e)
      );
    }, pollInterval);

    isPolling.value = true;
    console.log(`[TrinityStore] Polling started with ${pollInterval}ms interval`);
  }

  /**
   * Stop polling for Trinity updates.
   */
  function stopPolling(): void {
    if (pollingIntervalId !== null) {
      clearInterval(pollingIntervalId);
      pollingIntervalId = null;
    }

    isPolling.value = false;
    console.log('[TrinityStore] Polling stopped');
  }

  /**
   * Clear all stored data and errors.
   */
  function clearAll(): void {
    isAvailable.value = false;
    version.value = null;
    registryEntries.value = [];
    instances.value = [];
    events.value = [];
    lastUpdated.value = null;
    connectionError.value = null;

    errors.value = {
      status: null,
      registry: null,
      instances: null,
      events: null,
    };
  }

  /**
   * Connect to the Trinity runtime.
   * @returns Connection result with success status and optional error/sessionId
   */
  async function connect(): Promise<TrinityConnectionResult> {
    connectionError.value = null;

    try {
      const result = await connectTrinity();

      if (!result.success) {
        connectionError.value = result.error ?? 'Failed to connect to Trinity runtime';
        console.error('[TrinityStore] Connection failed:', connectionError.value);
      }

      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown connection error';
      connectionError.value = message;
      console.error('[TrinityStore] Connection failed:', message);
      return {
        success: false,
        error: message,
      };
    }
  }

  /**
   * Initialize the Trinity store.
   * First attempts to connect, then checks status and optionally starts polling.
   * @param autoStartPolling - Whether to start polling automatically
   */
  async function initialize(autoStartPolling = false): Promise<void> {
    console.log('[TrinityStore] Initializing...');

    try {
      // First attempt to connect to Trinity
      const connectionResult = await connect();

      if (!connectionResult.success) {
        console.warn('[TrinityStore] Connection failed, skipping initialization');
        isAvailable.value = false;
        return;
      }

      // Connection succeeded, check status
      await checkStatus();

      if (isAvailable.value) {
        await refreshAll();
      }

      // Only start polling if connection and status check succeeded
      if (autoStartPolling && isAvailable.value) {
        startPolling();
      }

      console.log('[TrinityStore] Initialization complete');
    } catch (error) {
      console.error('[TrinityStore] Initialization failed:', error);
    }
  }

  /**
   * Cleanup the Trinity store (stop polling, clear data).
   */
  function cleanup(): void {
    stopPolling();
    clearAll();
    console.log('[TrinityStore] Cleanup complete');
  }

  // ===========================================================================
  // Return Store
  // ===========================================================================

  return {
    // State
    isAvailable,
    version,
    registryEntries,
    instances,
    events,
    lastUpdated,
    loading,
    errors,
    connectionError,
    isPolling,

    // Computed
    componentCount,
    systemCount,
    resourceCount,
    eventCount,
    totalRegistryCount,
    totalInstanceCount,
    isLoading,
    hasErrors,
    getEntriesByType,
    getInstancesByComponent,

    // Actions
    connect,
    checkStatus,
    refreshRegistry,
    refreshInstances,
    refreshEvents,
    refreshAll,
    startPolling,
    stopPolling,
    clearAll,
    initialize,
    cleanup,
  };
});

// Re-export types for convenience
export type {
  TrinityStatus,
  TrinityConnectionResult,
  RegistryEntry,
  RegistryEntryType,
  TrinityInstance,
  TrinityEvent,
  RegistryContents,
  InstancesQueryResult,
  RecentEventsResult,
} from '@/bridge/trinity';
