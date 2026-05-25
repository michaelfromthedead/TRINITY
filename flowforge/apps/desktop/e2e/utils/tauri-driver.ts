import { _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { spawn, ChildProcess, execSync } from 'child_process'
import { resolve } from 'path'
import { existsSync } from 'fs'

/**
 * Tauri Driver utility for launching and controlling Tauri applications in E2E tests.
 *
 * This module provides:
 * - App launching (binary or dev mode)
 * - WebView page access
 * - App lifecycle management
 *
 * Note: Tauri 2.0 uses a WebView-based architecture, so we interact with it
 * similarly to how we'd interact with a browser page.
 */

export interface TauriAppContext {
  /** The spawned Tauri process */
  process: ChildProcess | null
  /** The Playwright page for the WebView */
  page: Page | null
  /** The dev server URL if running in dev mode */
  devServerUrl: string | null
  /** Whether the app is running in dev mode */
  isDevMode: boolean
}

/**
 * Launch options for the Tauri app
 */
export interface LaunchOptions {
  /** Path to the Tauri binary (optional, uses env or default) */
  binaryPath?: string
  /** Whether to run in dev mode (uses Vite dev server) */
  devMode?: boolean
  /** Environment variables to pass to the app */
  env?: Record<string, string>
  /** Timeout for app startup in ms */
  timeout?: number
}

/**
 * Default launch timeout
 */
const DEFAULT_TIMEOUT = 30000

/**
 * Dev server URL
 */
const DEV_SERVER_URL = 'http://localhost:1420'

/**
 * Launch the Tauri application for testing.
 *
 * In production mode, launches the compiled binary.
 * In dev mode, starts the Vite dev server and connects to it.
 */
export async function launchTauriApp(options: LaunchOptions = {}): Promise<TauriAppContext> {
  const {
    binaryPath = process.env.TAURI_BINARY_PATH,
    devMode = process.env.TAURI_DEV_MODE === 'true',
    env = {},
    timeout = DEFAULT_TIMEOUT,
  } = options

  const context: TauriAppContext = {
    process: null,
    page: null,
    devServerUrl: null,
    isDevMode: devMode,
  }

  if (devMode) {
    // Dev mode: start the Vite dev server and Tauri
    console.log('[Tauri Driver] Starting in dev mode...')

    const rootDir = resolve(__dirname, '../..')

    // Start Tauri dev (which also starts Vite)
    context.process = spawn('npm', ['run', 'tauri:dev'], {
      cwd: rootDir,
      stdio: 'pipe',
      shell: true,
      env: {
        ...process.env,
        ...env,
        // Disable GPU acceleration for headless testing
        WEBKIT_DISABLE_COMPOSITING_MODE: '1',
      },
    })

    context.devServerUrl = DEV_SERVER_URL

    // Wait for the dev server to be ready
    await waitForDevServer(DEV_SERVER_URL, timeout)

    console.log('[Tauri Driver] Dev server ready')
  } else if (binaryPath && existsSync(binaryPath)) {
    // Production mode: launch the binary directly
    console.log(`[Tauri Driver] Launching binary: ${binaryPath}`)

    context.process = spawn(binaryPath, [], {
      stdio: 'pipe',
      env: {
        ...process.env,
        ...env,
        // Disable GPU acceleration for headless testing
        WEBKIT_DISABLE_COMPOSITING_MODE: '1',
      },
    })

    // Wait for the app window to be ready
    await new Promise((resolve) => setTimeout(resolve, 3000))
  } else {
    throw new Error(
      'No Tauri binary found and dev mode not enabled. ' +
      'Either build the app with "npm run tauri:build" or set TAURI_DEV_MODE=true'
    )
  }

  return context
}

/**
 * Wait for the Vite dev server to be ready
 */
async function waitForDevServer(url: string, timeout: number): Promise<void> {
  const startTime = Date.now()

  while (Date.now() - startTime < timeout) {
    try {
      const response = await fetch(url, { method: 'HEAD' })
      if (response.ok) {
        return
      }
    } catch {
      // Server not ready yet
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }

  throw new Error(`Dev server at ${url} did not start within ${timeout}ms`)
}

/**
 * Close the Tauri application
 */
export async function closeTauriApp(context: TauriAppContext): Promise<void> {
  if (context.process) {
    console.log('[Tauri Driver] Closing app...')

    // Kill the process tree
    if (process.platform === 'win32') {
      try {
        execSync(`taskkill /pid ${context.process.pid} /T /F`, { stdio: 'ignore' })
      } catch {
        // Process may already be dead
      }
    } else {
      // Unix: kill process group
      try {
        process.kill(-context.process.pid!, 'SIGKILL')
      } catch {
        // Try direct kill
        context.process.kill('SIGKILL')
      }
    }

    context.process = null
  }

  if (context.page) {
    await context.page.close().catch(() => {})
    context.page = null
  }
}

/**
 * Get selectors for common FlowForge UI elements
 */
export const Selectors = {
  // Main layout
  appContainer: '.app-container',
  appHeader: '.app-header',
  appMain: '.app-main',
  appTitle: '.app-title',

  // Canvas
  canvasContainer: '.graph-canvas-container',
  canvas: '.litegraph-canvas',
  canvasLoading: '.canvas-loading',

  // Sidebar
  sidebarTab: (id: string) => `[data-tab-id="${id}"]`,
  nodePalette: '.node-palette',
  fileExplorer: '.file-explorer',
  trinitySidebar: '.trinity-sidebar',

  // Node Search
  nodeSearch: '.node-search',
  nodeSearchInput: '.node-search .search-input',
  nodeSearchResults: '.node-search .results-list',
  nodeSearchResultItem: '.node-search .result-item',
  nodeSearchClose: '.node-search .close-button',

  // Dialogs
  dialog: '.global-dialog',
  dialogOverlay: '.dialog-overlay',

  // Bottom panel
  bottomPanel: '.bottom-panel-content-wrapper',
  bottomPanelTab: (name: string) => `.tab-btn:has-text("${name}")`,

  // Header elements
  trinityStatus: '.trinity-status',
  inspectorToggle: '.header-btn[title*="Inspector"]',
  modifiedIndicator: '.modified-indicator',
  fileName: '.file-name',

  // Context menu
  contextMenu: '.node-context-menu',

  // Type filter
  typeFilter: '.type-filter-toolbar',
  typeFilterButton: (type: string) => `.type-filter-btn[data-type="${type}"]`,
} as const

/**
 * Keyboard shortcuts used in FlowForge
 */
export const Shortcuts = {
  // File operations
  newGraph: { key: 'n', modifiers: ['Control'] },
  openFile: { key: 'o', modifiers: ['Control'] },
  saveFile: { key: 's', modifiers: ['Control'] },
  saveFileAs: { key: 's', modifiers: ['Control', 'Shift'] },

  // Edit operations
  undo: { key: 'z', modifiers: ['Control'] },
  redo: { key: 'z', modifiers: ['Control', 'Shift'] },
  redoAlt: { key: 'y', modifiers: ['Control'] },
  delete: { key: 'Delete', modifiers: [] },
  selectAll: { key: 'a', modifiers: ['Control'] },

  // Search
  searchNodes: { key: 'f', modifiers: ['Control'] },

  // View
  zoomToFit: { key: '1', modifiers: ['Control'] },
  resetView: { key: '0', modifiers: ['Control'] },
  toggleFocusMode: { key: 'f', modifiers: ['Control', 'Shift'] },
  toggleInspector: { key: 'i', modifiers: ['Control', 'Shift'] },
} as const

/**
 * Helper to press a keyboard shortcut
 */
export async function pressShortcut(
  page: Page,
  shortcut: { key: string; modifiers: string[] }
): Promise<void> {
  const keys = [...shortcut.modifiers, shortcut.key].join('+')
  await page.keyboard.press(keys)
}
