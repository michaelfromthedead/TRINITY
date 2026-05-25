import { defineConfig, devices } from '@playwright/test'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

/**
 * E2E Test Timeouts - mirrors values from @/config/flowforge.config.ts TEST_CONFIG.e2e
 * Duplicated here because playwright.config.ts runs outside the app context
 */
const E2E_TIMEOUTS = {
  actionTimeout: 10000,
  navigationTimeout: 30000,
  globalTimeout: 60000,
  expectTimeout: 10000,
} as const

/**
 * Playwright configuration for FlowForge Desktop E2E testing with Tauri.
 *
 * This configuration is designed to test the Tauri application by:
 * 1. Building the Tauri app before tests (optional, for CI)
 * 2. Launching the app via the Tauri driver
 * 3. Running tests against the WebView window
 *
 * @see https://playwright.dev/docs/test-configuration
 * @see https://tauri.app/v1/guides/testing/webdriver/
 */

// Path to the Tauri binary (varies by platform)
const tauriBinaryPath = process.env.TAURI_BINARY_PATH ?? (() => {
  const platform = process.platform
  const binaryName = platform === 'win32' ? 'flowforge.exe' : 'flowforge'
  const targetDir = resolve(__dirname, '../src-tauri/target')

  // Check debug first, then release
  const debugPath = resolve(targetDir, 'debug', binaryName)
  const releasePath = resolve(targetDir, 'release', binaryName)

  // Default to debug build for development
  return debugPath
})()

export default defineConfig({
  // Test directory
  testDir: '.',
  testMatch: '**/*.spec.ts',

  // Run tests in files in parallel
  fullyParallel: false, // Tauri apps need sequential execution

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,

  // Retry on CI only
  retries: process.env.CI ? 2 : 0,

  // Only 1 worker for Tauri (single app instance)
  workers: 1,

  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['list'],
    ...(process.env.CI ? [['github'] as const] : []),
  ],

  // Shared settings for all projects
  use: {
    // Base URL for any relative navigation
    baseURL: 'tauri://localhost',

    // Collect trace on first retry
    trace: 'on-first-retry',

    // Screenshot on failure
    screenshot: 'only-on-failure',

    // Video on failure
    video: 'on-first-retry',

    // Default timeout for actions
    actionTimeout: E2E_TIMEOUTS.actionTimeout,

    // Default timeout for navigation
    navigationTimeout: E2E_TIMEOUTS.navigationTimeout,
  },

  // Configure projects for different testing scenarios
  projects: [
    {
      name: 'tauri-desktop',
      use: {
        ...devices['Desktop Chrome'],
        // Tauri uses WebKit on macOS, Chrome on Windows/Linux
        // We'll use a custom launch configuration
      },
    },
  ],

  // Global timeout for each test
  timeout: E2E_TIMEOUTS.globalTimeout,

  // Expect timeout
  expect: {
    timeout: E2E_TIMEOUTS.expectTimeout,
  },

  // Output folder for test artifacts
  outputDir: 'test-results',

  // Global setup/teardown (temporarily disabled for testing)
  // globalSetup: resolve(__dirname, 'setup/global-setup.ts'),
  // globalTeardown: resolve(__dirname, 'setup/global-teardown.ts'),
})

// Export binary path for use in tests
export { tauriBinaryPath }
