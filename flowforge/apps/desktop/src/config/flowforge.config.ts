/**
 * FlowForge Universal Configuration
 *
 * Central configuration for all magic numbers, colors, and settings.
 * Import this instead of hardcoding values throughout the codebase.
 */

// =============================================================================
// CANVAS CONFIGURATION
// =============================================================================

export const CANVAS_CONFIG = {
  /** Background color for the graph canvas */
  backgroundColor: '#1a1a2e',

  /** Minimum zoom scale */
  minScale: 0.1,

  /** Maximum zoom scale */
  maxScale: 2,

  /** Default zoom scale */
  defaultScale: 1,

  /** Grid size in pixels */
  gridSize: 20,

  /** Padding when zooming to fit all nodes */
  zoomFitPadding: 50,

  /** Padding around grouped nodes */
  groupPadding: 20,

  /** Default group color */
  defaultGroupColor: '#3b82f6',
} as const

// =============================================================================
// NODE CONFIGURATION
// =============================================================================

export const NODE_CONFIG = {
  /** Default node size [width, height] */
  defaultSize: [200, 100] as [number, number],

  /** Minimum node width */
  minWidth: 140,

  /** Minimum node height */
  minHeight: 60,

  /** Layout measurements */
  layout: {
    /** Height of the node title bar */
    titleHeight: 30,

    /** Height of each input/output slot */
    slotHeight: 20,

    /** Height of each field row */
    fieldHeight: 18,

    /** Internal padding */
    padding: 10,

    /** Method row height */
    methodHeight: 14,

    /** Payload label height */
    payloadLabelHeight: 14,

    /** Source file indicator height */
    sourceIndicatorHeight: 12,
  },

  /** Node type-specific sizes */
  sizes: {
    component: [200, 80] as [number, number],
    system: [220, 100] as [number, number],
    resource: [200, 80] as [number, number],
    event: [200, 80] as [number, number],
  },
} as const

// =============================================================================
// TRINITY THEME COLORS
// =============================================================================

export const TRINITY_COLORS = {
  component: {
    color: '#3b82f6',
    colorLight: '#60a5fa',
    colorDark: '#2563eb',
    bgcolor: 'rgba(59, 130, 246, 0.15)',
    borderColor: '#3b82f6',
  },
  system: {
    color: '#22c55e',
    colorLight: '#4ade80',
    colorDark: '#16a34a',
    bgcolor: 'rgba(34, 197, 94, 0.15)',
    borderColor: '#22c55e',
  },
  resource: {
    color: '#a855f7',
    colorLight: '#c084fc',
    colorDark: '#9333ea',
    bgcolor: 'rgba(168, 85, 247, 0.15)',
    borderColor: '#a855f7',
  },
  event: {
    color: '#f97316',
    colorLight: '#fb923c',
    colorDark: '#ea580c',
    bgcolor: 'rgba(249, 115, 22, 0.15)',
    borderColor: '#f97316',
  },
} as const

export type TrinityColorKey = keyof typeof TRINITY_COLORS

// =============================================================================
// FONTS
// =============================================================================

export const FONTS = {
  /** Monospace font for code/fields */
  code: '11px monospace',

  /** Small monospace font for methods */
  codeSmall: '10px monospace',

  /** Label font */
  label: '10px Arial',

  /** Small label font */
  labelSmall: '9px Arial',

  /** Bold small label */
  labelBold: 'bold 9px Arial',
} as const

// =============================================================================
// UI CONFIGURATION
// =============================================================================

export const UI_CONFIG = {
  /** History/undo configuration */
  history: {
    /** Maximum number of history entries */
    maxSize: 50,
  },

  /** Toast notification configuration */
  toast: {
    /** Default toast duration in ms */
    defaultDuration: 3000,
  },

  /** Dialog configuration */
  dialog: {
    /** Maximum number of stacked dialogs */
    maxStack: 10,
  },

  /** Auto-save configuration */
  autosave: {
    /** Auto-save interval in ms */
    interval: 60000,

    /** Whether auto-save is enabled by default */
    enabled: true,
  },

  /** Node search configuration */
  search: {
    /** Debounce time for search input (ms) */
    debounceMs: 150,
    /** Panel width */
    panelWidth: 320,
    /** Panel max height */
    panelMaxHeight: 400,
    /** Highlight color for matching nodes */
    highlightColor: '#fbbf24',
  },

  /** Source indicator configuration */
  sourceIndicator: {
    /** Maximum path display length */
    maxDisplayLength: 40,
    /** Feedback display duration (ms) */
    feedbackDuration: 2000,
  },

  /** Context menu configuration */
  contextMenu: {
    /** Default width of context menus */
    width: 280,
    /** Default height of context menus */
    height: 300,
    /** Margin from viewport edges */
    edgeMargin: 10,
  },

  /** Diff preview configuration */
  diff: {
    /** Number of context lines around changes */
    contextLines: 3,
    /** Backup file suffix */
    backupSuffix: '.bak',
  },

  /** File watcher configuration */
  fileWatcher: {
    /** File change polling interval in ms */
    pollInterval: 2000,
  },

  /** Event log configuration */
  eventLog: {
    /** Event polling interval in ms */
    pollInterval: 500,
  },
} as const

// =============================================================================
// TEXT COLORS
// =============================================================================

export const TEXT_COLORS = {
  /** Muted text color */
  muted: '#666680',

  /** Field name color */
  fieldName: '#a0a0a0',

  /** Source file indicator color */
  sourceFile: '#666680',
} as const

// =============================================================================
// EDITOR CONFIGURATION
// =============================================================================

export const EDITOR_CONFIG = {
  /** Common editor command templates */
  templates: {
    vscode: 'code --goto {file}:{line}',
    cursor: 'cursor --goto {file}:{line}',
    sublime: 'subl {file}:{line}',
    vim: 'vim +{line} {file}',
    neovim: 'nvim +{line} {file}',
    emacs: 'emacs +{line} {file}',
    kate: 'kate --line {line} {file}',
    gedit: 'gedit +{line} {file}',
  },
  /** Default editor command (empty = system default) */
  defaultCommand: '',
} as const

// =============================================================================
// TRINITY INTROSPECTION CONFIGURATION
// =============================================================================

export const TRINITY_CONFIG = {
  /** Polling interval for Trinity status updates (milliseconds) */
  pollingInterval: 2000,

  /** Maximum number of events to keep in memory */
  maxEvents: 100,

  /** Timeout for Trinity API calls (milliseconds) */
  apiTimeout: 5000,

  /** Whether to auto-start polling on store initialization */
  autoStartPolling: false,
} as const

// =============================================================================
// DEFAULT SETTINGS
// =============================================================================

export const DEFAULT_SETTINGS: Record<string, unknown> = {
  theme: 'dark',
  autoSave: UI_CONFIG.autosave.enabled,
  autoSaveInterval: UI_CONFIG.autosave.interval,
  showGrid: true,
  snapToGrid: true,
  gridSize: CANVAS_CONFIG.gridSize,
  // Editor settings
  editorCommand: EDITOR_CONFIG.defaultCommand,
}

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

export const TEST_CONFIG = {
  /** E2E test timeouts */
  e2e: {
    /** Action timeout in ms */
    actionTimeout: 10000,
    /** Navigation timeout in ms */
    navigationTimeout: 30000,
    /** Global test timeout in ms */
    globalTimeout: 60000,
    /** Short wait for UI updates (ms) */
    shortWait: 100,
    /** Medium wait for animations (ms) */
    mediumWait: 300,
    /** Long wait for operations (ms) */
    longWait: 500,
  },
  /** Integration test timeouts */
  integration: {
    /** IPC call timeout in ms */
    ipcTimeout: 10000,
    /** Sidecar startup timeout in ms */
    sidecarStartup: 5000,
  },
  /** Mock data defaults */
  mocks: {
    /** Default node position offset for layout */
    nodePositionOffset: 250,
  },
} as const

// =============================================================================
// COMBINED CONFIG EXPORT
// =============================================================================

export const CONFIG = {
  canvas: CANVAS_CONFIG,
  node: NODE_CONFIG,
  colors: TRINITY_COLORS,
  fonts: FONTS,
  ui: UI_CONFIG,
  text: TEXT_COLORS,
  editor: EDITOR_CONFIG,
  trinity: TRINITY_CONFIG,
  test: TEST_CONFIG,
  defaultSettings: DEFAULT_SETTINGS,
} as const

export default CONFIG
