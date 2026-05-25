import { FullConfig } from '@playwright/test'

/**
 * Global teardown for Playwright E2E tests with Tauri.
 *
 * This teardown:
 * 1. Cleans up any lingering processes
 * 2. Generates summary reports if needed
 */

async function globalTeardown(config: FullConfig): Promise<void> {
  console.log('\n[E2E Teardown] Starting global teardown...')

  // Any cleanup tasks can be added here
  // For example, killing lingering Tauri processes

  if (process.platform !== 'win32') {
    // On Unix-like systems, kill any lingering flowforge processes
    const { exec } = await import('child_process')
    exec('pkill -f flowforge 2>/dev/null || true', (err) => {
      // Ignore errors - process may not exist
    })
  }

  console.log('[E2E Teardown] Global teardown complete\n')
}

export default globalTeardown
