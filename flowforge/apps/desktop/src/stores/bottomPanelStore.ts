/**
 * Bottom Panel Store - Bottom panel/terminal management for FlowForge
 * Extracted from ComfyUI frontend, stripped of SD-specific code
 */
import { defineStore } from 'pinia'
import { computed, ref, type Component } from 'vue'

import { useCommandStore } from './commandStore'

export type PanelType = 'terminal' | 'output' | 'problems' | 'console'

export interface BottomPanelExtension {
  id: string
  title?: string
  titleKey?: string
  icon?: string
  component?: Component
  targetPanel?: PanelType
  order?: number
}

interface PanelState {
  tabs: BottomPanelExtension[]
  activeTabId: string
  visible: boolean
}

export const useBottomPanelStore = defineStore('bottomPanel', () => {
  // Multi-panel state
  const panels = ref<Record<PanelType, PanelState>>({
    terminal: { tabs: [], activeTabId: '', visible: false },
    output: { tabs: [], activeTabId: '', visible: false },
    problems: { tabs: [], activeTabId: '', visible: false },
    console: { tabs: [], activeTabId: '', visible: false }
  })

  const activePanel = ref<PanelType | null>(null)

  // Computed properties for active panel
  const activePanelState = computed(() =>
    activePanel.value ? panels.value[activePanel.value] : null
  )

  const activeBottomPanelTab = computed<BottomPanelExtension | null>(() => {
    const state = activePanelState.value
    if (!state) return null
    return state.tabs.find((tab) => tab.id === state.activeTabId) ?? null
  })

  const bottomPanelVisible = computed({
    get: () => !!activePanel.value,
    set: (visible: boolean) => {
      if (!visible) {
        activePanel.value = null
      }
    }
  })

  const bottomPanelTabs = computed(() => activePanelState.value?.tabs ?? [])

  const activeBottomPanelTabId = computed({
    get: () => activePanelState.value?.activeTabId ?? '',
    set: (tabId: string) => {
      const state = activePanelState.value
      if (state) {
        state.activeTabId = tabId
      }
    }
  })

  const togglePanel = (panelType: PanelType) => {
    const panel = panels.value[panelType]
    if (panel.tabs.length === 0) return

    if (activePanel.value === panelType) {
      // Hide current panel
      activePanel.value = null
    } else {
      // Show target panel
      activePanel.value = panelType
      if (!panel.activeTabId && panel.tabs.length > 0) {
        panel.activeTabId = panel.tabs[0].id
      }
    }
  }

  const toggleBottomPanel = () => {
    // Default to terminal panel
    togglePanel('terminal')
  }

  const setActiveTab = (tabId: string) => {
    const state = activePanelState.value
    if (state) {
      state.activeTabId = tabId
    }
  }

  const toggleBottomPanelTab = (tabId: string) => {
    // Find which panel contains this tab
    for (const [panelType, panel] of Object.entries(panels.value)) {
      const tab = panel.tabs.find((t) => t.id === tabId)
      if (tab) {
        if (activePanel.value === panelType && panel.activeTabId === tabId) {
          activePanel.value = null
        } else {
          activePanel.value = panelType as PanelType
          panel.activeTabId = tabId
        }
        return
      }
    }
  }

  const registerBottomPanelTab = (tab: BottomPanelExtension) => {
    const targetPanel = tab.targetPanel ?? 'terminal'
    const panel = panels.value[targetPanel]

    // Avoid duplicates
    if (panel.tabs.some((t) => t.id === tab.id)) {
      console.warn(`Bottom panel tab ${tab.id} already registered`)
      return
    }

    panel.tabs = [...panel.tabs, tab]
    if (panel.tabs.length === 1) {
      panel.activeTabId = tab.id
    }

    const tabName = tab.title || tab.titleKey || tab.id
    useCommandStore().registerCommand({
      id: `Workspace.ToggleBottomPanelTab.${tab.id}`,
      icon: tab.icon || 'pi pi-list',
      label: `Toggle ${tabName} Panel`,
      category: 'view-controls',
      function: () => toggleBottomPanelTab(tab.id),
      source: 'System'
    })
  }

  const unregisterBottomPanelTab = (tabId: string) => {
    for (const panel of Object.values(panels.value)) {
      const index = panel.tabs.findIndex((t) => t.id === tabId)
      if (index !== -1) {
        panel.tabs = panel.tabs.filter((t) => t.id !== tabId)
        if (panel.activeTabId === tabId) {
          panel.activeTabId = panel.tabs[0]?.id ?? ''
        }
        return
      }
    }
  }

  /**
   * Register the core bottom panel tabs for FlowForge
   */
  const registerCoreBottomPanelTabs = () => {
    // Terminal tab
    registerBottomPanelTab({
      id: 'terminal',
      title: 'Terminal',
      icon: 'pi pi-code',
      targetPanel: 'terminal'
    })

    // Output tab
    registerBottomPanelTab({
      id: 'output',
      title: 'Output',
      icon: 'pi pi-file',
      targetPanel: 'output'
    })

    // Problems tab
    registerBottomPanelTab({
      id: 'problems',
      title: 'Problems',
      icon: 'pi pi-exclamation-triangle',
      targetPanel: 'problems'
    })

    // Console tab
    registerBottomPanelTab({
      id: 'console',
      title: 'Console',
      icon: 'pi pi-desktop',
      targetPanel: 'console'
    })
  }

  return {
    // Multi-panel API
    panels,
    activePanel,
    togglePanel,

    bottomPanelVisible,
    toggleBottomPanel,
    bottomPanelTabs,
    activeBottomPanelTab,
    activeBottomPanelTabId,
    setActiveTab,
    toggleBottomPanelTab,
    registerBottomPanelTab,
    unregisterBottomPanelTab,
    registerCoreBottomPanelTabs
  }
})
