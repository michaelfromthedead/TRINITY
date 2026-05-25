/**
 * FlowForge Services
 *
 * Exports the API interface and a factory function that returns
 * the appropriate implementation based on the runtime environment.
 */

// Export types
export type {
  FlowForgeAPI,
  NodeGraph,
  GraphNode,
  GraphEdge,
  OpenPythonFileResult,
  TrinityNodeTypes,
  NodeTypeDefinition,
  NodePortDefinition,
  NodePropertyDefinition,
  ExecutionResponse,
  ExecutionStatus,
  AppInfo,
} from './api';

// Export implementations
export { TauriAPI, tauriApi } from './tauriApi';
export { MockAPI, mockApi, createMockNode, createMockEdge, createEmptyMockGraph } from './mockApi';

// Import for factory function
import type { FlowForgeAPI } from './api';
import { TauriAPI } from './tauriApi';
import { MockAPI } from './mockApi';

// =============================================================================
// ENVIRONMENT DETECTION
// =============================================================================

/**
 * Check if running in Tauri environment.
 */
export function isTauri(): boolean {
  // Tauri v2 injects __TAURI_INTERNALS__ (and optionally __TAURI__ if withGlobalTauri is true)
  return typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);
}

/**
 * Check if running in development mode.
 * Uses Vite's import.meta.env.DEV which is injected at build time.
 */
export function isDevelopment(): boolean {
  return import.meta.env?.DEV ?? false;
}

/**
 * Check if running in browser without Tauri.
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined' && !isTauri();
}

// =============================================================================
// API FACTORY
// =============================================================================

/**
 * Cached API instance.
 */
let cachedApi: FlowForgeAPI | null = null;

/**
 * API factory options.
 */
export interface ApiFactoryOptions {
  /**
   * Force using a specific API implementation.
   * - 'tauri': Always use TauriAPI (will fail in browser)
   * - 'mock': Always use MockAPI
   * - 'auto': Detect environment (default)
   */
  mode?: 'tauri' | 'mock' | 'auto';

  /**
   * Simulated delay for mock API (milliseconds).
   * Only applies when using MockAPI.
   */
  mockDelay?: number;

  /**
   * Whether to cache the API instance.
   * Default: true
   */
  cache?: boolean;
}

/**
 * Get the FlowForge API instance.
 *
 * Returns TauriAPI in desktop environment, MockAPI in browser.
 *
 * @example
 * ```typescript
 * const api = getApi();
 * const nodes = await api.getObjectInfo();
 * ```
 */
export function getApi(options: ApiFactoryOptions = {}): FlowForgeAPI {
  const { mode = 'auto', mockDelay, cache = true } = options;

  // Return cached instance if available
  if (cache && cachedApi !== null) {
    return cachedApi;
  }

  let api: FlowForgeAPI;

  switch (mode) {
    case 'tauri':
      api = new TauriAPI();
      break;

    case 'mock':
      api = mockDelay !== undefined
        ? new MockAPI({ simulatedDelay: mockDelay })
        : new MockAPI();
      break;

    case 'auto':
    default:
      if (isTauri()) {
        api = new TauriAPI();
        console.log('[FlowForge] Using TauriAPI (desktop mode)');
      } else {
        api = new MockAPI({ simulatedDelay: mockDelay ?? 100 });
        console.log('[FlowForge] Using MockAPI (browser development mode)');
      }
      break;
  }

  // Cache the instance
  if (cache) {
    cachedApi = api;
  }

  return api;
}

/**
 * Clear the cached API instance.
 * Useful for testing or switching implementations.
 */
export function clearApiCache(): void {
  cachedApi = null;
}

/**
 * Create a new API instance without caching.
 * Useful for testing or when you need multiple instances.
 */
export function createApi(options: ApiFactoryOptions = {}): FlowForgeAPI {
  return getApi({ ...options, cache: false });
}

// =============================================================================
// DEFAULT EXPORT
// =============================================================================

/**
 * Default API instance using auto-detection.
 * This is a lazy singleton that initializes on first use.
 */
const api: FlowForgeAPI = new Proxy({} as FlowForgeAPI, {
  get(_target, prop: keyof FlowForgeAPI) {
    const instance = getApi();
    const value = instance[prop];
    if (typeof value === 'function') {
      return value.bind(instance);
    }
    return value;
  },
});

export default api;
