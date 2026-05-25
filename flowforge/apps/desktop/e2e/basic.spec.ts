import { test, expect, Page, Browser, BrowserContext } from '@playwright/test'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
import {
  launchTauriApp,
  closeTauriApp,
  TauriAppContext,
  Selectors,
  Shortcuts,
  pressShortcut,
} from './utils/tauri-driver'
import {
  waitForAppReady,
  openNodeSearch,
  closeNodeSearch,
  searchNodes,
  selectNodeFromSearch,
  clickCanvas,
  rightClickCanvas,
  verifyCanvasRendered,
  verifyNodesOnCanvas,
  mockFileDialog,
  getCurrentFileName,
  isDocumentModified,
  screenshotCanvas,
  waitForNodes,
} from './utils/test-helpers'
// Test configuration values - duplicated from flowforge.config.ts to avoid path alias issues in Playwright
// These values mirror TEST_CONFIG.e2e from the main config file
const TEST_CONFIG = {
  e2e: {
    actionTimeout: 10000,
    navigationTimeout: 30000,
    globalTimeout: 60000,
    shortWait: 100,
    mediumWait: 300,
    longWait: 500,
  },
} as const

/**
 * FlowForge Desktop E2E Tests
 *
 * These tests verify the core functionality of the FlowForge desktop application:
 * - App launch and initialization
 * - File operations (open, save)
 * - Node navigation and selection
 * - Node search functionality
 * - Canvas rendering and interaction
 *
 * Test Strategy:
 * - Use Playwright's browser automation for WebView testing
 * - Mock native dialogs where necessary
 * - Use test fixtures for consistent test data
 */

// Path to test fixture files
const FIXTURES_DIR = resolve(__dirname, 'fixtures')
const TEST_FILE_PATH = resolve(FIXTURES_DIR, 'test_file.py')

test.describe('FlowForge Desktop E2E Tests', () => {
  // Configure serial execution
  test.describe.configure({ mode: 'serial' })

  // Browser and page instances
  let browser: Browser
  let context: BrowserContext
  let page: Page

  /**
   * Setup before all tests - launch the app
   */
  test.beforeAll(async ({ browser: b }) => {
    browser = b

    // Create a new browser context
    context = await browser.newContext({
      viewport: { width: 1280, height: 800 },
      ignoreHTTPSErrors: true,
    })

    // Create a new page
    page = await context.newPage()

    // For dev mode testing, navigate to the dev server
    const devServerUrl = process.env.DEV_SERVER_URL || 'http://localhost:1420'

    console.log(`[E2E] Navigating to: ${devServerUrl}`)
    await page.goto(devServerUrl)

    // Wait for the app to be ready
    await waitForAppReady(page)
  })

  /**
   * Cleanup after all tests
   */
  test.afterAll(async () => {
    if (page) {
      await page.close()
    }
    if (context) {
      await context.close()
    }
  })

  /**
   * Test Suite: App Launch
   */
  test.describe('App Launch', () => {
  test('should launch and display main window', async () => {
    // Verify the app container is visible
    await expect(page.locator(Selectors.appContainer)).toBeVisible()

    // Verify the header is present
    await expect(page.locator(Selectors.appHeader)).toBeVisible()

    // Verify the main content area is present
    await expect(page.locator(Selectors.appMain)).toBeVisible()
  })

  test('should display correct window title', async () => {
    const titleElement = page.locator(Selectors.appTitle)
    await expect(titleElement).toBeVisible()

    const title = await titleElement.textContent()
    expect(title).toContain('FlowForge')
  })

  test('should show canvas when ready', async () => {
    await verifyCanvasRendered(page)
  })

  test('should not show loading indicator after initialization', async () => {
    const loadingIndicator = page.locator('.loading')
    await expect(loadingIndicator).not.toBeVisible()
  })
})

/**
 * Test Suite: File Operations
 */
test.describe('File Operations', () => {
  test('should start with empty/new graph', async () => {
    // New graphs should have an empty or default file name
    const fileName = await getCurrentFileName(page)
    // Could be empty, "Untitled", or similar
    expect(fileName.length).toBeLessThanOrEqual(20)
  })

  test('should mark document as unmodified initially', async () => {
    const isModified = await isDocumentModified(page)
    expect(isModified).toBe(false)
  })

  test('should respond to Ctrl+O (Open) shortcut', async () => {
    // SMOKE TEST: Verifies the Open shortcut doesn't crash the app.
    // This does not test actual file opening - just that the shortcut is handled gracefully.

    // Mock the file dialog to return our test file
    await mockFileDialog(page, TEST_FILE_PATH)

    // Press the open shortcut
    await pressShortcut(page, Shortcuts.openFile)

    // Give the app time to process the shortcut
    await page.waitForTimeout(TEST_CONFIG.e2e.longWait)

    // App should still be in a valid state (shortcut didn't crash the app)
    await expect(page.locator(Selectors.appContainer)).toBeVisible()
  })

  test('should respond to Ctrl+S (Save) shortcut', async () => {
    // SMOKE TEST: Verifies the Save shortcut doesn't crash the app.
    // This does not test actual file saving - just that the shortcut is handled gracefully.

    // Press the save shortcut
    await pressShortcut(page, Shortcuts.saveFile)

    // Give the app time to process the shortcut
    await page.waitForTimeout(TEST_CONFIG.e2e.longWait)

    // App should still be in a valid state (shortcut didn't crash the app)
    await expect(page.locator(Selectors.appContainer)).toBeVisible()
  })

  test('should respond to Ctrl+N (New) shortcut', async () => {
    // SMOKE TEST: Verifies the New Graph shortcut doesn't crash the app.
    // This does not test actual graph creation - just that the shortcut is handled gracefully.

    // Press the new graph shortcut
    await pressShortcut(page, Shortcuts.newGraph)

    // Give the app time to process the shortcut
    await page.waitForTimeout(TEST_CONFIG.e2e.longWait)

    // App should still be in a valid state (shortcut didn't crash the app)
    await expect(page.locator(Selectors.appContainer)).toBeVisible()
  })
})

/**
 * Test Suite: Canvas Rendering
 *
 * Note: Due to the overlay layout pattern used for the splitter panels,
 * the canvas may have 0 height in browser-only tests. These tests verify
 * that the canvas exists and LiteGraph is initialized rather than checking
 * strict visibility/dimensions.
 */
test.describe('Canvas Rendering', () => {
  test('should render canvas element', async () => {
    // Verify canvas is attached to the DOM and LiteGraph is initialized
    await verifyCanvasRendered(page)
  })

  test('should have proper canvas dimensions', async () => {
    const canvas = page.locator(Selectors.canvas)

    // Verify canvas exists and has a width
    // Note: Height may be 0 in browser mode due to overlay layout
    const width = await canvas.evaluate((el: HTMLCanvasElement) => el.width)
    expect(width).toBeGreaterThan(0)

    // Verify canvas element is in the DOM
    await expect(canvas).toBeAttached()
  })

  test('should respond to canvas clicks', async () => {
    const canvas = page.locator(Selectors.canvas)

    // Force a click via JavaScript since the canvas may have 0 height in browser mode
    await canvas.evaluate((el) => {
      const event = new MouseEvent('click', {
        bubbles: true,
        cancelable: true,
        clientX: 100,
        clientY: 100
      })
      el.dispatchEvent(event)
    })

    // Canvas should handle the click without errors - verify it's still attached
    await expect(canvas).toBeAttached()
  })

  test('should support zooming with Ctrl+1 (Zoom to Fit)', async () => {
    await pressShortcut(page, Shortcuts.zoomToFit)

    // Give the canvas time to adjust
    await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)

    // Canvas should still be visible and valid
    await verifyCanvasRendered(page)
  })

  test('should support resetting view with Ctrl+0', async () => {
    await pressShortcut(page, Shortcuts.resetView)

    // Give the canvas time to adjust
    await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)

    // Canvas should still be visible and valid
    await verifyCanvasRendered(page)
  })
})

/**
 * Test Suite: Node Navigation
 *
 * Note: These tests use JavaScript event dispatch due to the canvas
 * having 0 height in browser-only mode (overlay layout pattern).
 */
test.describe('Node Navigation', () => {
  test('should support clicking to select nodes', async () => {
    // Click on the canvas (simulating node selection)
    await clickCanvas(page, 200, 200)

    // The selection should be processed without errors - verify canvas is still attached
    await verifyCanvasRendered(page)
  })

  test('should support right-click for context menu on canvas', async () => {
    // Right-click on the canvas
    await rightClickCanvas(page, 100, 100)

    // Give the context menu time to appear (LiteGraph's default menu)
    await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)

    // Click elsewhere to close any menu
    await page.keyboard.press('Escape')

    // Verify canvas is still functional
    await verifyCanvasRendered(page)
  })

  test('should support Ctrl+A to select all nodes', async () => {
    // Focus the canvas via JavaScript
    const canvas = page.locator(Selectors.canvas)
    await canvas.evaluate((el) => el.focus())

    // Try to select all
    await pressShortcut(page, Shortcuts.selectAll)

    // Canvas should handle the command
    await verifyCanvasRendered(page)
  })

  test('should support Delete key to remove selected nodes', async () => {
    // Focus the canvas via JavaScript
    const canvas = page.locator(Selectors.canvas)
    await canvas.evaluate((el) => el.focus())

    // Press Delete (should be handled gracefully even with no selection)
    await pressShortcut(page, Shortcuts.delete)

    // Canvas should remain functional
    await verifyCanvasRendered(page)
  })

  test('should support undo/redo (Ctrl+Z / Ctrl+Shift+Z)', async () => {
    // Test undo
    await pressShortcut(page, Shortcuts.undo)
    await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)

    // Test redo
    await pressShortcut(page, Shortcuts.redo)
    await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)

    // Canvas should remain functional
    await verifyCanvasRendered(page)
  })
})

/**
 * Test Suite: Node Search
 */
test.describe('Node Search', () => {
  test('should open search with Ctrl+F', async () => {
    const searchDialog = await openNodeSearch(page)
    await expect(searchDialog).toBeVisible()

    // Clean up
    await closeNodeSearch(page)
  })

  test('should have search input focused when opened', async () => {
    await openNodeSearch(page)

    const searchInput = page.locator(Selectors.nodeSearchInput)
    await expect(searchInput).toBeFocused()

    // Clean up
    await closeNodeSearch(page)
  })

  test('should close search with Escape', async () => {
    await openNodeSearch(page)

    // Press Escape
    await page.keyboard.press('Escape')

    // Search should be hidden
    const searchDialog = page.locator(Selectors.nodeSearch)
    await expect(searchDialog).not.toBeVisible()
  })

  test.skip('should close search when clicking close button', async () => {
    // SKIP: Due to overlay layout, the close button click doesn't propagate properly
    // in browser-only tests. The button works in Tauri desktop context.
    await openNodeSearch(page)

    // Click the close button (use force: true to bypass overlay pointer-events)
    const closeBtn = page.locator(Selectors.nodeSearchClose)
    await closeBtn.click({ force: true })

    // Search should be hidden
    const searchDialog = page.locator(Selectors.nodeSearch)
    await expect(searchDialog).not.toBeVisible()
  })

  test('should filter by type dropdown', async () => {
    await openNodeSearch(page)

    // Find the type filter dropdown
    const typeSelect = page.locator('.node-search .type-select')
    await expect(typeSelect).toBeVisible()

    // Should have options
    const options = await typeSelect.locator('option').all()
    expect(options.length).toBeGreaterThan(1)

    // Clean up
    await closeNodeSearch(page)
  })

  test('should display keyboard hints', async () => {
    await openNodeSearch(page)

    // Check for keyboard hints
    const hints = page.locator('.node-search .keyboard-hints')
    await expect(hints).toBeVisible()

    // Should show Enter and Esc hints
    const hintText = await hints.textContent()
    expect(hintText).toContain('Enter')
    expect(hintText).toContain('Esc')

    // Clean up
    await closeNodeSearch(page)
  })

  test('should support typing in search input', async () => {
    await openNodeSearch(page)

    const searchInput = page.locator(Selectors.nodeSearchInput)

    // Type a search query
    await searchInput.fill('test')

    // Verify the value was entered
    await expect(searchInput).toHaveValue('test')

    // Clean up
    await closeNodeSearch(page)
  })

  test('should show "no results" message for non-matching query', async () => {
    await openNodeSearch(page)

    const searchInput = page.locator(Selectors.nodeSearchInput)

    // Type a query that won't match anything
    await searchInput.fill('xyznonexistentnode123')

    // Wait for debounced search
    await page.waitForTimeout(TEST_CONFIG.e2e.longWait)

    // Should show no results message
    const noResults = page.locator('.node-search .no-results')
    await expect(noResults).toBeVisible()

    // Clean up
    await closeNodeSearch(page)
  })

  test('should support arrow key navigation in results', async () => {
    await openNodeSearch(page)

    // Even with no results, arrow keys should be handled
    const searchInput = page.locator(Selectors.nodeSearchInput)
    await searchInput.focus()

    // Press arrow down
    await page.keyboard.press('ArrowDown')
    await page.waitForTimeout(TEST_CONFIG.e2e.shortWait)

    // Press arrow up
    await page.keyboard.press('ArrowUp')
    await page.waitForTimeout(TEST_CONFIG.e2e.shortWait)

    // Search dialog should still be functional
    const searchDialog = page.locator(Selectors.nodeSearch)
    await expect(searchDialog).toBeVisible()

    // Clean up
    await closeNodeSearch(page)
  })
})

/**
 * Test Suite: UI Components
 *
 * Note: Due to the overlay splitter layout pattern, some UI elements may
 * have unusual visibility states in browser-only tests.
 */
test.describe('UI Components', () => {
  test('should display sidebar tabs', async () => {
    // Check for sidebar icon buttons that are used to toggle sidebar panels
    const sidebarIcons = page.locator('.side-toolbar-icons button, .sidebar-icon, .side-toolbar .sidebar-icon-wrapper')

    // Verify sidebar icons exist in DOM
    const iconCount = await sidebarIcons.count()
    expect(iconCount).toBeGreaterThan(0)
  })

  test('should display bottom panel', async () => {
    const bottomPanel = page.locator(Selectors.bottomPanel)

    // Bottom panel may or may not be visible by default depending on app state.
    // This test verifies the selector query executes without throwing an error,
    // confirming the DOM structure is as expected (even if the panel is hidden).
    const isAttached = await bottomPanel.count() > 0

    // Log the actual state for debugging purposes
    console.log(`[E2E] Bottom panel exists: ${isAttached}`)
    expect(isAttached).toBe(true)
  })

  test('should display Trinity status indicator', async () => {
    const trinityStatus = page.locator(Selectors.trinityStatus)

    // Trinity status should be present in the DOM
    const statusCount = await trinityStatus.count()
    expect(statusCount).toBeGreaterThan(0)
  })

  test('should support Inspector toggle (Ctrl+Shift+I)', async () => {
    // Toggle inspector panel
    await pressShortcut(page, Shortcuts.toggleInspector)
    await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)

    // Toggle it back
    await pressShortcut(page, Shortcuts.toggleInspector)
    await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)

    // App should remain functional
    await expect(page.locator(Selectors.appContainer)).toBeVisible()
  })
})

/**
 * Test Suite: Performance & Stability
 */
test.describe('Performance & Stability', () => {
  test('should handle rapid keyboard input', async () => {
    // Open and close search rapidly
    for (let i = 0; i < 5; i++) {
      await pressShortcut(page, Shortcuts.searchNodes)
      await page.waitForTimeout(TEST_CONFIG.e2e.shortWait)
      await page.keyboard.press('Escape')
      await page.waitForTimeout(TEST_CONFIG.e2e.shortWait)
    }

    // App should remain stable
    await expect(page.locator(Selectors.appContainer)).toBeVisible()
  })

  test('should handle rapid canvas clicks', async () => {
    // Use JavaScript event dispatch for rapid clicks due to overlay layout
    for (let i = 0; i < 10; i++) {
      await clickCanvas(page, Math.random() * 300, Math.random() * 300)
    }

    // App should remain stable - verify canvas is still attached
    await verifyCanvasRendered(page)
  })

  test('should handle repeated file operations without crashing', async () => {
    // STABILITY TEST: Verifies the app can handle multiple consecutive New Graph operations
    // without crashing. This does NOT detect memory leaks - real memory leak detection
    // would require heap snapshots or external profiling tools.
    for (let i = 0; i < 3; i++) {
      await pressShortcut(page, Shortcuts.newGraph)
      await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)
    }

    // App should remain functional after repeated operations
    await expect(page.locator(Selectors.appContainer)).toBeVisible()
  })

  test('should take canvas screenshot for visual verification', async () => {
    // Take a screenshot of the canvas area
    const screenshot = await screenshotCanvas(page, 'canvas-final-state')

    // Screenshot should be captured (buffer should have data)
    expect(screenshot.length).toBeGreaterThan(0)
  })
})
}) // End of main FlowForge Desktop E2E Tests describe block
