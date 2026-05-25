/**
 * Menu Item Store - Menu bar management for FlowForge
 * Extracted from ComfyUI frontend, stripped of SD-specific code
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'

import { useCommandStore } from './commandStore'
import type { FlowForgeCommandImpl } from './commandStore'

export interface MenuItem {
  label?: string
  key?: string
  icon?: string
  tooltip?: string
  command?: () => void | Promise<void>
  items?: MenuItem[]
  separator?: boolean
  disabled?: boolean
  visible?: boolean
  flowForgeCommand?: FlowForgeCommandImpl
  parentPath?: string
}

export const useMenuItemStore = defineStore('menuItem', () => {
  const commandStore = useCommandStore()
  const menuItems = ref<MenuItem[]>([])
  const menuItemHasActiveStateChildren = ref<Record<string, boolean>>({})

  const registerMenuGroup = (path: string[], items: MenuItem[]) => {
    let currentLevel = menuItems.value

    // Traverse the path, creating nodes if necessary
    for (let i = 0; i < path.length; i++) {
      const segment = path[i]
      let found = currentLevel.find((item) => item.label === segment)

      if (!found) {
        // Create a new node if it doesn't exist
        found = {
          label: segment,
          key: segment,
          items: []
        }
        currentLevel.push(found)
      }

      // Ensure the found item has an 'items' array
      if (!found.items) {
        found.items = []
      }

      // Move to the next level
      currentLevel = found.items
    }

    if (currentLevel.length > 0) {
      currentLevel.push({
        separator: true
      })
    }
    // Add the new items to the last level
    currentLevel.push(...items)

    // Store if any of the children have active state as we will hide the icon if they do
    const parentPath = path.join('.')
    if (!menuItemHasActiveStateChildren.value[parentPath]) {
      menuItemHasActiveStateChildren.value[parentPath] = items.some(
        (item) => item.flowForgeCommand?.active
      )
    }
  }

  const registerCommands = (path: string[], commandIds: string[]) => {
    const items = commandIds
      .map((commandId) => commandStore.getCommand(commandId))
      .filter((command) => command !== undefined)
      .map(
        (command) =>
          ({
            command: () => commandStore.execute(command.id),
            label: command.menubarLabel,
            icon: command.icon,
            tooltip: command.tooltip,
            flowForgeCommand: command,
            parentPath: path.join('.')
          }) as MenuItem
      )
    registerMenuGroup(path, items)
  }

  /**
   * Load menu commands from an extension/plugin
   */
  const loadExtensionMenuCommands = (extension: {
    name: string
    commands?: { id: string }[]
    menuCommands?: { path: string[]; commands: string[] }[]
  }) => {
    if (!extension.menuCommands) {
      return
    }

    const extensionCommandIds = new Set(
      extension.commands?.map((command) => command.id) ?? []
    )
    extension.menuCommands.forEach((menuCommand) => {
      const commands = menuCommand.commands.filter((command) =>
        extensionCommandIds.has(command)
      )
      if (commands.length) {
        registerCommands(menuCommand.path, commands)
      }
    })
  }

  /**
   * Register core menu commands for FlowForge
   */
  const registerCoreMenuCommands = () => {
    // File menu - with proper structure including separators
    // First group: New and Open
    registerCommands(['File'], [
      'FlowForge.NewFile',
      'FlowForge.OpenFile',
    ])

    // Second group: Save operations
    registerCommands(['File'], [
      'FlowForge.SaveFile',
      'FlowForge.SaveFileAs',
    ])

    // Third group: Export
    registerCommands(['File'], [
      'FlowForge.Export',
    ])

    // Fourth group: Exit (separate from other operations)
    registerCommands(['File'], [
      'FlowForge.Exit',
    ])

    // Edit menu
    registerCommands(['Edit'], [
      'FlowForge.Undo',
      'FlowForge.Redo',
      'FlowForge.Cut',
      'FlowForge.Copy',
      'FlowForge.Paste',
      'FlowForge.SelectAll'
    ])

    // View menu
    registerCommands(['View'], [
      'FlowForge.ZoomIn',
      'FlowForge.ZoomOut',
      'FlowForge.FitView',
      'FlowForge.ToggleSidebar',
      'FlowForge.ToggleBottomPanel'
    ])

    // Graph menu
    registerCommands(['Graph'], [
      'FlowForge.AddNode',
      'FlowForge.DeleteSelected',
      'FlowForge.GroupSelected',
      'FlowForge.UngroupSelected'
    ])
  }

  const clearMenuItems = () => {
    menuItems.value = []
    menuItemHasActiveStateChildren.value = {}
  }

  return {
    menuItems,
    registerMenuGroup,
    registerCommands,
    loadExtensionMenuCommands,
    registerCoreMenuCommands,
    menuItemHasActiveStateChildren,
    clearMenuItems
  }
})
