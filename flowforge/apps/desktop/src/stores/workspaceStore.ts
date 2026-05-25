/**
 * Workspace Store - Central workspace state for FlowForge
 * Extracted from ComfyUI frontend, stripped of SD-specific code
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { useBottomPanelStore } from './bottomPanelStore'
import { useCommandStore } from './commandStore'
import { useDialogStore } from './dialogStore'
import { useGraphStore } from './graphStore'
import { useSidebarTabStore } from './sidebarTabStore'
import type { SidebarTabExtension } from './sidebarTabStore'
import { DEFAULT_SETTINGS, UI_CONFIG } from '@/config/flowforge.config'

export interface ToastMessage {
  severity: 'success' | 'info' | 'warn' | 'error'
  summary: string
  detail?: string
  life?: number
}

export const useWorkspaceStore = defineStore('workspace', () => {
  // UI State
  const spinner = ref(false)
  const focusMode = ref(false)
  const shiftDown = ref(false)

  // Toast notifications
  const toasts = ref<ToastMessage[]>([])

  // Access to other stores
  const command = computed(() => ({
    commands: useCommandStore().commands,
    execute: useCommandStore().execute
  }))

  const sidebarTab = computed(() => useSidebarTabStore())
  const bottomPanel = computed(() => useBottomPanelStore())
  const dialog = computed(() => useDialogStore())
  const graph = computed(() => useGraphStore())

  // Settings (simplified version - will be expanded later)
  const settings = ref<Record<string, unknown>>({ ...DEFAULT_SETTINGS })

  const setting = computed(() => ({
    settings: settings.value,
    get: (key: string) => settings.value[key],
    set: (key: string, value: unknown) => {
      settings.value[key] = value
    }
  }))

  // Toast management
  function addToast(toast: ToastMessage) {
    toasts.value.push(toast)
    if (toast.life !== 0) {
      setTimeout(() => {
        removeToast(toast)
      }, toast.life ?? UI_CONFIG.toast.defaultDuration)
    }
  }

  function removeToast(toast: ToastMessage) {
    const index = toasts.value.indexOf(toast)
    if (index > -1) {
      toasts.value.splice(index, 1)
    }
  }

  function clearToasts() {
    toasts.value = []
  }

  // Focus mode
  function toggleFocusMode() {
    focusMode.value = !focusMode.value
  }

  // Sidebar tab management (backwards compatibility)
  function registerSidebarTab(tab: SidebarTabExtension) {
    sidebarTab.value.registerSidebarTab(tab)
  }

  function unregisterSidebarTab(id: string) {
    sidebarTab.value.unregisterSidebarTab(id)
  }

  function getSidebarTabs(): SidebarTabExtension[] {
    return sidebarTab.value.sidebarTabs
  }

  // Keyboard state tracking
  function setShiftDown(value: boolean) {
    shiftDown.value = value
  }

  // Spinner/loading state
  function showSpinner() {
    spinner.value = true
  }

  function hideSpinner() {
    spinner.value = false
  }

  async function withSpinner<T>(fn: () => Promise<T>): Promise<T> {
    showSpinner()
    try {
      return await fn()
    } finally {
      hideSpinner()
    }
  }

  return {
    // UI State
    spinner,
    focusMode,
    shiftDown,
    toasts,

    // Store access
    command,
    sidebarTab,
    bottomPanel,
    dialog,
    graph,
    setting,

    // Toast methods
    addToast,
    removeToast,
    clearToasts,

    // Focus mode
    toggleFocusMode,

    // Sidebar methods (backwards compatibility)
    registerSidebarTab,
    unregisterSidebarTab,
    getSidebarTabs,

    // Keyboard
    setShiftDown,

    // Spinner
    showSpinner,
    hideSpinner,
    withSpinner
  }
})
