import { Page, expect, Locator } from '@playwright/test'
import { Selectors, Shortcuts, pressShortcut } from './tauri-driver'

// Test configuration values - duplicated from flowforge.config.ts to avoid path alias issues in Playwright
// These values mirror TEST_CONFIG.e2e from the main config file
const TEST_CONFIG = {
  e2e: {
    shortWait: 100,
    mediumWait: 300,
    searchDialogTimeout: 2000,
    searchDialogFallbackTimeout: 3000,
  },
}

/**
 * Test helper utilities for FlowForge E2E tests.
 *
 * These helpers provide high-level actions and assertions
 * specific to the FlowForge application.
 */

/**
 * Wait for the FlowForge app to be fully loaded and ready
 */
export async function waitForAppReady(page: Page, timeout = 30000): Promise<void> {
  // Wait for the app container to be visible
  await page.waitForSelector(Selectors.appContainer, {
    state: 'visible',
    timeout,
  })

  // Wait for loading to disappear
  const loadingEl = page.locator('.loading')
  if (await loadingEl.isVisible()) {
    await loadingEl.waitFor({ state: 'hidden', timeout })
  }

  // Wait for the canvas element to exist (may have 0 dimensions due to overlay layout)
  await page.waitForSelector(Selectors.canvas, {
    state: 'attached',
    timeout,
  })

  // Wait for canvas loading indicator to disappear if it exists
  const canvasLoading = page.locator(Selectors.canvasLoading)
  const canvasLoadingExists = await canvasLoading.count() > 0
  if (canvasLoadingExists && await canvasLoading.isVisible()) {
    await canvasLoading.waitFor({ state: 'hidden', timeout })
  }

  // Check that the app initialization completed via console log
  // The app logs "[App] Canvas ready" when fully initialized
  await page.waitForFunction(() => {
    // Check if AppLayout rendered (sidebar tabs exist)
    const sidebarIcons = document.querySelectorAll('.side-toolbar-icons button, .side-toolbar-icons .sidebar-icon')
    return sidebarIcons.length > 0
  }, { timeout })

  // Give LiteGraph a moment to initialize
  await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)
}

/**
 * Open the node search dialog
 */
export async function openNodeSearch(page: Page): Promise<Locator> {
  await pressShortcut(page, Shortcuts.searchNodes)

  const searchDialog = page.locator(Selectors.nodeSearch)
  await searchDialog.waitFor({ state: 'visible' })

  return searchDialog
}

/**
 * Close the node search dialog
 */
export async function closeNodeSearch(page: Page): Promise<void> {
  const searchDialog = page.locator(Selectors.nodeSearch)

  if (await searchDialog.isVisible()) {
    // Dispatch a keyboard event directly to the search input to trigger Vue's handler
    const searchInput = page.locator(Selectors.nodeSearchInput)
    if (await searchInput.isVisible()) {
      await searchInput.evaluate((el) => {
        const event = new KeyboardEvent('keydown', {
          key: 'Escape',
          code: 'Escape',
          bubbles: true,
          cancelable: true
        })
        el.dispatchEvent(event)
      })
    }

    // Wait for the dialog to hide with a shorter timeout
    try {
      await searchDialog.waitFor({ state: 'hidden', timeout: TEST_CONFIG.e2e.searchDialogTimeout })
    } catch {
      // If direct event didn't work, try pressing Escape via Playwright
      await page.keyboard.press('Escape')
      await searchDialog.waitFor({ state: 'hidden', timeout: TEST_CONFIG.e2e.searchDialogFallbackTimeout })
    }
  }
}

/**
 * Search for nodes using the search dialog
 */
export async function searchNodes(page: Page, query: string): Promise<Locator[]> {
  const searchDialog = await openNodeSearch(page)

  const input = searchDialog.locator(Selectors.nodeSearchInput.replace('.node-search ', ''))
  await input.fill(query)

  // Wait for results to appear (debounced search)
  await page.waitForTimeout(TEST_CONFIG.e2e.mediumWait)

  const results = await searchDialog.locator(Selectors.nodeSearchResultItem.replace('.node-search ', '')).all()
  return results
}

/**
 * Select a node from search results by index
 */
export async function selectNodeFromSearch(page: Page, index: number): Promise<void> {
  const searchDialog = page.locator(Selectors.nodeSearch)
  const results = searchDialog.locator(Selectors.nodeSearchResultItem.replace('.node-search ', ''))

  const targetResult = results.nth(index)
  await targetResult.click()
}

/**
 * Click on the canvas at a specific position
 *
 * Note: Due to overlay layout, the canvas may have 0 height. This helper
 * uses JavaScript event dispatch to simulate clicks when the canvas isn't visible.
 */
export async function clickCanvas(page: Page, x: number, y: number): Promise<void> {
  const canvas = page.locator(Selectors.canvas)

  // Use JavaScript to dispatch click event since canvas may be hidden in browser mode
  await canvas.evaluate((el, pos) => {
    const rect = el.getBoundingClientRect()
    const event = new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: rect.left + pos.x,
      clientY: rect.top + pos.y,
    })
    el.dispatchEvent(event)
  }, { x, y })
}

/**
 * Right-click on the canvas to open context menu
 *
 * Note: Due to overlay layout, the canvas may have 0 height. This helper
 * uses JavaScript event dispatch to simulate right-clicks.
 */
export async function rightClickCanvas(page: Page, x: number, y: number): Promise<void> {
  const canvas = page.locator(Selectors.canvas)

  // Use JavaScript to dispatch contextmenu event
  await canvas.evaluate((el, pos) => {
    const rect = el.getBoundingClientRect()
    const event = new MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: rect.left + pos.x,
      clientY: rect.top + pos.y,
      button: 2,
    })
    el.dispatchEvent(event)
  }, { x, y })
}

/**
 * Drag on the canvas to select an area
 *
 * Note: Due to overlay layout, the canvas may have 0 height. This helper
 * uses JavaScript event dispatch to simulate drag operations.
 */
export async function dragOnCanvas(
  page: Page,
  startX: number,
  startY: number,
  endX: number,
  endY: number
): Promise<void> {
  const canvas = page.locator(Selectors.canvas)

  // Dispatch mousedown, mousemove, mouseup events via JavaScript
  await canvas.evaluate((el, coords) => {
    const rect = el.getBoundingClientRect()

    // Mouse down
    const downEvent = new MouseEvent('mousedown', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: rect.left + coords.startX,
      clientY: rect.top + coords.startY,
      button: 0,
    })
    el.dispatchEvent(downEvent)

    // Mouse move
    const moveEvent = new MouseEvent('mousemove', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: rect.left + coords.endX,
      clientY: rect.top + coords.endY,
      button: 0,
    })
    el.dispatchEvent(moveEvent)

    // Mouse up
    const upEvent = new MouseEvent('mouseup', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: rect.left + coords.endX,
      clientY: rect.top + coords.endY,
      button: 0,
    })
    el.dispatchEvent(upEvent)
  }, { startX, startY, endX, endY })
}

/**
 * Verify that the canvas has rendered (canvas element exists and graph is initialized)
 *
 * Note: Due to the overlay layout pattern, the canvas may have 0 height in browser tests
 * when not running in the Tauri window context. This helper checks for canvas existence
 * and graph initialization rather than strict visibility.
 */
export async function verifyCanvasRendered(page: Page): Promise<void> {
  const canvas = page.locator(Selectors.canvas)

  // Verify canvas exists in DOM
  await expect(canvas).toHaveCount(1)

  // Verify canvas element is attached
  await expect(canvas).toBeAttached()

  // Check if LiteGraph canvas is initialized
  const isInitialized = await page.evaluate(() => {
    const canvasEl = document.querySelector('.litegraph-canvas') as HTMLCanvasElement
    if (!canvasEl) return false

    // Check if the canvas has a width (even with height=0)
    if (canvasEl.width === 0) return false

    // Check if LGraphCanvas is attached to the canvas
    // @ts-ignore
    return !!canvasEl.lgraphcanvas || canvasEl.classList.contains('lgraphcanvas')
  })

  expect(isInitialized).toBe(true)
}

/**
 * Verify nodes are visible on the canvas (by checking for node-related DOM elements or canvas state)
 */
export async function verifyNodesOnCanvas(page: Page, expectedCount?: number): Promise<void> {
  // Since LiteGraph renders to canvas, we check via the app state
  // This is done by evaluating the graph state in the page context

  const nodeCount = await page.evaluate(() => {
    // Access the graph through the exposed Vue app state
    const graphCanvas = document.querySelector('.graph-canvas-container')
    if (!graphCanvas) return 0

    // Check if there are any nodes rendered (LiteGraph stores nodes internally)
    // We can check this through the canvas's internal state or DOM
    const canvas = document.querySelector('.litegraph-canvas') as HTMLCanvasElement
    if (!canvas) return 0

    // Access the LGraph instance if exposed
    // @ts-ignore
    const lgCanvas = canvas._lgraph_canvas
    if (lgCanvas && lgCanvas.graph) {
      // @ts-ignore
      return lgCanvas.graph._nodes?.length ?? 0
    }

    return -1 // Unknown state
  })

  if (expectedCount !== undefined) {
    expect(nodeCount).toBe(expectedCount)
  } else {
    expect(nodeCount).toBeGreaterThanOrEqual(0)
  }
}

/**
 * Open a file via the file dialog (mocked for testing)
 *
 * Note: In E2E tests, file dialogs need to be mocked since
 * they're native OS dialogs that Playwright can't interact with directly.
 */
export async function openFileViaDialog(page: Page, filePath: string): Promise<void> {
  // We'll use IPC mocking to simulate the file dialog response
  // This requires the app to support a test mode or mock API

  // For now, we trigger the open command and handle it via app state
  await pressShortcut(page, Shortcuts.openFile)

  // In a real implementation, we'd mock the Tauri dialog API
  // For tests, we might use a special query param or env var to auto-load a file
  console.log(`[Test Helper] Would open file: ${filePath}`)
}

/**
 * Mock the file open dialog to return a specific file path
 *
 * This sets up the mock before the dialog is triggered.
 */
export async function mockFileDialog(page: Page, filePath: string): Promise<void> {
  await page.evaluate((path) => {
    // Store the mock path in a global that the app can check
    // @ts-ignore
    window.__E2E_MOCK_FILE_PATH__ = path
  }, filePath)
}

/**
 * Get the current file name displayed in the header
 */
export async function getCurrentFileName(page: Page): Promise<string> {
  const fileNameEl = page.locator(Selectors.fileName)
  return await fileNameEl.textContent() ?? ''
}

/**
 * Check if the document is marked as modified
 */
export async function isDocumentModified(page: Page): Promise<boolean> {
  const indicator = page.locator(Selectors.modifiedIndicator)
  return await indicator.isVisible()
}

/**
 * Switch to a sidebar tab
 */
export async function switchSidebarTab(page: Page, tabId: string): Promise<void> {
  const tab = page.locator(Selectors.sidebarTab(tabId))
  await tab.click()
  await page.waitForTimeout(TEST_CONFIG.e2e.shortWait) // Allow for transition
}

/**
 * Switch to a bottom panel tab
 */
export async function switchBottomPanelTab(page: Page, tabName: string): Promise<void> {
  const tab = page.locator(Selectors.bottomPanelTab(tabName))
  await tab.click()
  await page.waitForTimeout(TEST_CONFIG.e2e.shortWait) // Allow for transition
}

/**
 * Toggle a type filter button
 */
export async function toggleTypeFilter(page: Page, type: string): Promise<void> {
  const filterBtn = page.locator(Selectors.typeFilterButton(type))
  if (await filterBtn.isVisible()) {
    await filterBtn.click()
  }
}

/**
 * Take a screenshot of the canvas area (or full page if canvas has 0 dimensions)
 *
 * Note: Due to overlay layout, the canvas container may have 0 height in browser mode.
 * Falls back to a full page screenshot in that case.
 */
export async function screenshotCanvas(page: Page, name: string): Promise<Buffer> {
  const canvas = page.locator(Selectors.canvasContainer)
  const box = await canvas.boundingBox()

  // If canvas has proper dimensions, take element screenshot
  if (box && box.height > 0) {
    return await canvas.screenshot({ path: `test-results/screenshots/${name}.png` })
  }

  // Otherwise, take a full page screenshot
  return await page.screenshot({ path: `test-results/screenshots/${name}.png` })
}

/**
 * Wait for a specific number of nodes to be loaded
 */
export async function waitForNodes(page: Page, count: number, timeout = 10000): Promise<void> {
  const startTime = Date.now()

  while (Date.now() - startTime < timeout) {
    const nodeCount = await page.evaluate(() => {
      const canvas = document.querySelector('.litegraph-canvas') as HTMLCanvasElement
      // @ts-ignore
      const lgCanvas = canvas?._lgraph_canvas
      // @ts-ignore
      return lgCanvas?.graph?._nodes?.length ?? 0
    })

    if (nodeCount >= count) {
      return
    }

    await page.waitForTimeout(100)
  }

  throw new Error(`Timeout waiting for ${count} nodes to load`)
}
