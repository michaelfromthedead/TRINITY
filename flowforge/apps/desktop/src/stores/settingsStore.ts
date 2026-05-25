/**
 * Settings Store - Application settings management for FlowForge
 *
 * Manages user preferences including editor configuration.
 */
import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { DEFAULT_SETTINGS, EDITOR_CONFIG } from '@/config/flowforge.config'
import { setEditorCommand, detectEditors, type EditorInfo } from '@/bridge/editor'

/**
 * Settings interface for type safety.
 */
export interface FlowForgeSettings {
  theme: 'dark' | 'light'
  autoSave: boolean
  autoSaveInterval: number
  showGrid: boolean
  snapToGrid: boolean
  gridSize: number
  editorCommand: string
}

/**
 * Local storage key for persisting settings.
 */
const STORAGE_KEY = 'flowforge_settings'

/**
 * Load settings from local storage.
 */
function loadSettings(): FlowForgeSettings {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const parsed = JSON.parse(stored)
      return { ...DEFAULT_SETTINGS, ...parsed } as FlowForgeSettings
    }
  } catch (error) {
    console.warn('[Settings] Failed to load settings from storage:', error)
  }
  return DEFAULT_SETTINGS as FlowForgeSettings
}

/**
 * Save settings to local storage.
 */
function saveSettings(settings: FlowForgeSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
  } catch (error) {
    console.warn('[Settings] Failed to save settings to storage:', error)
  }
}

export const useSettingsStore = defineStore('settings', () => {
  // Settings state
  const settings = ref<FlowForgeSettings>(loadSettings())

  // Available editors (detected on the system)
  const availableEditors = ref<EditorInfo[]>([])

  // Individual setting accessors
  const theme = computed(() => settings.value.theme)
  const autoSave = computed(() => settings.value.autoSave)
  const autoSaveInterval = computed(() => settings.value.autoSaveInterval)
  const showGrid = computed(() => settings.value.showGrid)
  const snapToGrid = computed(() => settings.value.snapToGrid)
  const gridSize = computed(() => settings.value.gridSize)
  const editorCommand = computed(() => settings.value.editorCommand)

  // Watch for settings changes and persist
  watch(
    settings,
    (newSettings) => {
      saveSettings(newSettings)
      // Update the editor bridge with new command
      setEditorCommand(newSettings.editorCommand || null)
    },
    { deep: true }
  )

  /**
   * Get a setting value by key.
   */
  function get<K extends keyof FlowForgeSettings>(key: K): FlowForgeSettings[K] {
    return settings.value[key]
  }

  /**
   * Set a setting value by key.
   */
  function set<K extends keyof FlowForgeSettings>(
    key: K,
    value: FlowForgeSettings[K]
  ): void {
    settings.value[key] = value
  }

  /**
   * Set the editor command.
   */
  function setEditor(command: string): void {
    settings.value.editorCommand = command
  }

  /**
   * Reset settings to defaults.
   */
  function reset(): void {
    settings.value = { ...DEFAULT_SETTINGS } as FlowForgeSettings
  }

  /**
   * Reset editor command to default.
   */
  function resetEditorCommand(): void {
    settings.value.editorCommand = EDITOR_CONFIG.defaultCommand
  }

  /**
   * Detect available editors on the system.
   */
  async function detectAvailableEditors(): Promise<EditorInfo[]> {
    try {
      const editors = await detectEditors()
      availableEditors.value = editors
      return editors
    } catch (error) {
      console.warn('[Settings] Failed to detect editors:', error)
      availableEditors.value = [
        { name: 'System Default', command: '', detected: true },
      ]
      return availableEditors.value
    }
  }

  /**
   * Initialize the settings store.
   * Should be called on app startup.
   */
  async function initialize(): Promise<void> {
    // Set the editor command in the bridge
    setEditorCommand(settings.value.editorCommand || null)

    // Detect available editors
    await detectAvailableEditors()

    console.log('[Settings] Settings store initialized')
  }

  return {
    // State
    settings,
    availableEditors,

    // Computed accessors
    theme,
    autoSave,
    autoSaveInterval,
    showGrid,
    snapToGrid,
    gridSize,
    editorCommand,

    // Methods
    get,
    set,
    setEditor,
    reset,
    resetEditorCommand,
    detectAvailableEditors,
    initialize,
  }
})
