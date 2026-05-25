/**
 * Command Store - Command palette and action management for FlowForge
 * Extracted from ComfyUI frontend, stripped of SD-specific code
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { useKeybindingStore } from './keybindingStore'
import type { KeybindingImpl } from './keybindingStore'

// File operations are lazily imported to avoid circular dependencies
let fileOperationsInstance: ReturnType<typeof import('../composables/useFileOperations').useFileOperations> | null = null

async function getFileOperations() {
  if (!fileOperationsInstance) {
    const { useFileOperations } = await import('../composables/useFileOperations')
    fileOperationsInstance = useFileOperations()
  }
  return fileOperationsInstance
}

export interface FlowForgeCommand {
  id: string
  function: (metadata?: Record<string, unknown>) => void | Promise<void>

  label?: string | (() => string)
  icon?: string | (() => string)
  tooltip?: string | (() => string)
  menubarLabel?: string | (() => string)
  versionAdded?: string
  confirmation?: string
  source?: string
  active?: () => boolean
  category?: 'essentials' | 'view-controls' | 'graph' | 'file' | 'edit'
}

export class FlowForgeCommandImpl implements FlowForgeCommand {
  id: string
  function: (metadata?: Record<string, unknown>) => void | Promise<void>
  _label?: string | (() => string)
  _icon?: string | (() => string)
  _tooltip?: string | (() => string)
  _menubarLabel?: string | (() => string)
  versionAdded?: string
  confirmation?: string
  source?: string
  active?: () => boolean
  category?: 'essentials' | 'view-controls' | 'graph' | 'file' | 'edit'

  constructor(command: FlowForgeCommand) {
    this.id = command.id
    this.function = command.function
    this._label = command.label
    this._icon = command.icon
    this._tooltip = command.tooltip
    this._menubarLabel = command.menubarLabel ?? command.label
    this.versionAdded = command.versionAdded
    this.confirmation = command.confirmation
    this.source = command.source
    this.active = command.active
    this.category = command.category
  }

  get label() {
    return typeof this._label === 'function' ? this._label() : this._label
  }

  get icon() {
    return typeof this._icon === 'function' ? this._icon() : this._icon
  }

  get tooltip() {
    return typeof this._tooltip === 'function' ? this._tooltip() : this._tooltip
  }

  get menubarLabel() {
    return typeof this._menubarLabel === 'function'
      ? this._menubarLabel()
      : this._menubarLabel
  }

  get keybinding(): KeybindingImpl | null {
    return useKeybindingStore().getKeybindingByCommandId(this.id)
  }
}

export const useCommandStore = defineStore('command', () => {
  const commandsById = ref<Record<string, FlowForgeCommandImpl>>({})
  const commands = computed(() => Object.values(commandsById.value))

  const registerCommand = (command: FlowForgeCommand) => {
    if (commandsById.value[command.id]) {
      console.warn(`Command ${command.id} already registered`)
    }
    commandsById.value[command.id] = new FlowForgeCommandImpl(command)
  }

  const registerCommands = (commands: FlowForgeCommand[]) => {
    for (const command of commands) {
      registerCommand(command)
    }
  }

  const getCommand = (command: string) => {
    return commandsById.value[command]
  }

  const execute = async (
    commandId: string,
    options?: {
      errorHandler?: (error: unknown) => void
      metadata?: Record<string, unknown>
    }
  ) => {
    const command = getCommand(commandId)
    if (command) {
      try {
        await command.function(options?.metadata)
      } catch (error) {
        if (options?.errorHandler) {
          options.errorHandler(error)
        } else {
          console.error(`Error executing command ${commandId}:`, error)
          throw error
        }
      }
    } else {
      throw new Error(`Command ${commandId} not found`)
    }
  }

  const isRegistered = (command: string) => {
    return !!commandsById.value[command]
  }

  /**
   * Load commands from an extension/plugin
   */
  const loadExtensionCommands = (
    extension: { name: string; commands?: FlowForgeCommand[] }
  ) => {
    if (extension.commands) {
      for (const command of extension.commands) {
        registerCommand({
          ...command,
          source: extension.name
        })
      }
    }
  }

  const formatKeySequence = (command: FlowForgeCommandImpl): string => {
    const sequences = command.keybinding?.combo.getKeySequences() || []
    return sequences
      .map((seq) => seq.replace(/Control/g, 'Ctrl').replace(/Shift/g, 'Shift'))
      .join(' + ')
  }

  /**
   * Register core file commands for FlowForge
   */
  const registerCoreFileCommands = () => {
    registerCommands([
      {
        id: 'FlowForge.NewFile',
        label: 'New',
        menubarLabel: 'New',
        icon: 'file-plus',
        tooltip: 'Create a new file',
        category: 'file',
        function: async () => {
          const ops = await getFileOperations()
          await ops.newFile()
        },
      },
      {
        id: 'FlowForge.OpenFile',
        label: 'Open Python File...',
        menubarLabel: 'Open...',
        icon: 'folder-open',
        tooltip: 'Open a Python file',
        category: 'file',
        function: async () => {
          const ops = await getFileOperations()
          await ops.openFile()
        },
      },
      {
        id: 'FlowForge.SaveFile',
        label: 'Save',
        menubarLabel: 'Save',
        icon: 'save',
        tooltip: 'Save the current file',
        category: 'file',
        function: async () => {
          const ops = await getFileOperations()
          await ops.saveFile()
        },
      },
      {
        id: 'FlowForge.SaveFileAs',
        label: 'Save As...',
        menubarLabel: 'Save As...',
        icon: 'save',
        tooltip: 'Save the file with a new name',
        category: 'file',
        function: async () => {
          const ops = await getFileOperations()
          await ops.saveFileAs()
        },
      },
      {
        id: 'FlowForge.Export',
        label: 'Export',
        menubarLabel: 'Export',
        icon: 'download',
        tooltip: 'Export the workflow',
        category: 'file',
        function: async () => {
          // Placeholder for export functionality
          console.log('Export not yet implemented')
        },
      },
      {
        id: 'FlowForge.Exit',
        label: 'Exit',
        menubarLabel: 'Exit',
        icon: 'log-out',
        tooltip: 'Exit the application',
        category: 'file',
        function: async () => {
          // Check for unsaved changes before exit
          const ops = await getFileOperations()
          const canClose = await ops.confirmUnsavedChanges()
          if (canClose) {
            // In Tauri, we would call window close
            // For now, just log
            console.log('Exit requested')
          }
        },
      },
    ])
  }

  return {
    commands,
    execute,
    getCommand,
    registerCommand,
    registerCommands,
    isRegistered,
    loadExtensionCommands,
    formatKeySequence,
    registerCoreFileCommands,
  }
})
