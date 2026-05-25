/**
 * Vitest Test Setup
 *
 * Global setup for all tests in the FlowForge desktop app.
 * Mocks Tauri APIs and provides common test utilities.
 */

import { vi, beforeEach, afterEach } from 'vitest';

// =============================================================================
// TAURI API MOCKS
// =============================================================================

/**
 * Mock invoke function for Tauri IPC calls.
 * Tests can customize responses by using vi.mocked(invoke).mockResolvedValue()
 */
const mockInvoke = vi.fn();

// Mock @tauri-apps/api/core module
vi.mock('@tauri-apps/api/core', () => ({
  invoke: mockInvoke,
}));

// Make mockInvoke available globally for tests
declare global {
  // eslint-disable-next-line no-var
  var mockTauriInvoke: typeof mockInvoke;
}

globalThis.mockTauriInvoke = mockInvoke;

// =============================================================================
// WINDOW MOCKS
// =============================================================================

// Mock window.__TAURI__ to simulate Tauri environment when needed
Object.defineProperty(window, '__TAURI__', {
  value: undefined,
  writable: true,
  configurable: true,
});

// =============================================================================
// CONSOLE MOCKS
// =============================================================================

// Suppress console output during tests unless explicitly enabled
const originalConsole = { ...console };

beforeEach(() => {
  // Reset all mocks before each test
  vi.clearAllMocks();
  mockInvoke.mockReset();

  // Suppress console.log and console.warn in tests
  // Keep console.error for debugging test failures
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  // Restore console after each test
  console.log = originalConsole.log;
  console.warn = originalConsole.warn;
});

// =============================================================================
// TEST UTILITIES
// =============================================================================

/**
 * Helper to enable console output for debugging specific tests.
 */
export function enableConsoleOutput(): void {
  console.log = originalConsole.log;
  console.warn = originalConsole.warn;
}

/**
 * Helper to wait for all pending promises to resolve.
 */
export function flushPromises(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Helper to advance timers and flush promises.
 */
export async function advanceTimersAndFlush(ms: number): Promise<void> {
  vi.advanceTimersByTime(ms);
  await flushPromises();
}

// =============================================================================
// EXPORT MOCK FOR TEST FILES
// =============================================================================

export { mockInvoke };
