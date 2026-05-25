/**
 * Sidebar Tab Store - Sidebar panel management for FlowForge
 * Extracted from ComfyUI frontend, stripped of SD-specific code
 */
import { defineStore } from 'pinia'
import { computed, ref, type Component } from 'vue'

import { useCommandStore } from './commandStore'

export interface SidebarTabExtension {
  id: string
  title: string
  icon: string | Component
  tooltip?: string
  type: 'vue' | 'custom'
  component?: Component
  render?: (container: HTMLElement) => void
  destroy?: () => void
  order?: number
}

export const useSidebarTabStore = defineStore('sidebarTab', () => {
  const sidebarTabs = ref<SidebarTabExtension[]>([])
  const activeSidebarTabId = ref<string | null>(null)

  const activeSidebarTab = computed<SidebarTabExtension | null>(() => {
    return (
      sidebarTabs.value.find((tab) => tab.id === activeSidebarTabId.value) ??
      null
    )
  })

  const sortedSidebarTabs = computed(() => {
    return [...sidebarTabs.value].sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
  })

  const toggleSidebarTab = (tabId: string) => {
    activeSidebarTabId.value = activeSidebarTabId.value === tabId ? null : tabId
  }

  const registerSidebarTab = (tab: SidebarTabExtension) => {
    // Avoid duplicates
    if (sidebarTabs.value.some((t) => t.id === tab.id)) {
      console.warn(`Sidebar tab ${tab.id} already registered`)
      return
    }

    sidebarTabs.value = [...sidebarTabs.value, tab]

    // Register command to toggle this sidebar tab
    const commandStore = useCommandStore()
    commandStore.registerCommand({
      id: `Workspace.ToggleSidebarTab.${tab.id}`,
      icon: typeof tab.icon === 'string' ? tab.icon : undefined,
      label: () => `Toggle ${tab.title} Sidebar`,
      tooltip: tab.tooltip,
      category: 'view-controls',
      function: () => toggleSidebarTab(tab.id),
      active: () => activeSidebarTab.value?.id === tab.id,
      source: 'System'
    })
  }

  const unregisterSidebarTab = (id: string) => {
    const index = sidebarTabs.value.findIndex((tab) => tab.id === id)
    if (index !== -1) {
      const tab = sidebarTabs.value[index]
      if (tab.type === 'custom' && tab.destroy) {
        tab.destroy()
      }
      const newSidebarTabs = [...sidebarTabs.value]
      newSidebarTabs.splice(index, 1)
      sidebarTabs.value = newSidebarTabs

      // If the removed tab was active, clear the active tab
      if (activeSidebarTabId.value === id) {
        activeSidebarTabId.value = null
      }
    }
  }

  /**
   * Register the core sidebar tabs for FlowForge
   */
  const registerCoreSidebarTabs = () => {
    // These will be implemented with actual components
    // For now, register placeholder tabs
  }

  const setActiveTab = (tabId: string | null) => {
    activeSidebarTabId.value = tabId
  }

  return {
    sidebarTabs,
    sortedSidebarTabs,
    activeSidebarTabId,
    activeSidebarTab,
    toggleSidebarTab,
    registerSidebarTab,
    unregisterSidebarTab,
    registerCoreSidebarTabs,
    setActiveTab
  }
})
