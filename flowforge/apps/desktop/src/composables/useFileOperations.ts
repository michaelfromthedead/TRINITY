/**
 * File Operations Composable
 *
 * Provides file operations (new, open, save, save as) for FlowForge.
 * Handles unsaved changes confirmation and integrates with graphStore.
 */
import { ref } from 'vue'
import { useGraphStore } from '../stores/graphStore'
import { useDialogStore } from '../stores/dialogStore'
import { useWorkspaceStore } from '../stores/workspaceStore'
import {
  openPythonFile,
  savePythonFileDialog,
  writePythonFile,
} from '../bridge/files'
import type { GraphState, GraphNode } from '../stores/graphStore'
import { useRecentFiles } from './useRecentFiles'

export interface FileOperationsOptions {
  /** Custom confirmation handler. If not provided, uses dialogStore. */
  onConfirm?: (message: string) => Promise<boolean>
}

/**
 * Creates file operations composable.
 */
export function useFileOperations(options?: FileOperationsOptions) {
  const graphStore = useGraphStore()
  const dialogStore = useDialogStore()
  const workspaceStore = useWorkspaceStore()
  const { addRecentFile } = useRecentFiles()
  const isLoading = ref(false)
  const lastError = ref<string | null>(null)

  /**
   * Shows a confirmation dialog for unsaved changes.
   * Returns true if user wants to proceed, false otherwise.
   */
  async function confirmUnsavedChanges(): Promise<boolean> {
    if (!graphStore.isModified) {
      return true
    }

    if (options?.onConfirm) {
      return options.onConfirm(
        'You have unsaved changes. Do you want to discard them?'
      )
    }

    // Use native confirm for now - can be replaced with dialogStore modal later
    return window.confirm(
      'You have unsaved changes. Do you want to discard them?'
    )
  }

  /**
   * Shows a save confirmation dialog.
   * Returns 'save' | 'discard' | 'cancel'
   */
  async function confirmSaveBeforeAction(): Promise<'save' | 'discard' | 'cancel'> {
    if (!graphStore.isModified) {
      return 'discard' // No changes, proceed without saving
    }

    // For now, use a simple prompt. This can be enhanced with a proper dialog.
    const result = window.confirm(
      'Do you want to save your changes before proceeding?\n\nClick OK to save, Cancel to discard changes.'
    )

    if (result) {
      return 'save'
    }

    // Ask if they want to discard
    const discard = window.confirm('Discard unsaved changes?')
    return discard ? 'discard' : 'cancel'
  }

  /**
   * Creates a new file, clearing the current graph.
   */
  async function newFile(): Promise<boolean> {
    lastError.value = null

    // Check for unsaved changes
    const saveChoice = await confirmSaveBeforeAction()
    if (saveChoice === 'cancel') {
      return false
    }

    if (saveChoice === 'save') {
      const saved = await saveFile()
      if (!saved) {
        return false // Save was cancelled or failed
      }
    }

    // Clear the graph
    graphStore.clearGraph()
    return true
  }

  /**
   * Opens a Python file via native dialog.
   * Parses the Python file using the API and loads nodes into the graph.
   */
  async function openFile(): Promise<boolean> {
    lastError.value = null

    // Check for unsaved changes
    const saveChoice = await confirmSaveBeforeAction()
    if (saveChoice === 'cancel') {
      return false
    }

    if (saveChoice === 'save') {
      const saved = await saveFile()
      if (!saved) {
        return false // Save was cancelled or failed
      }
    }

    try {
      isLoading.value = true

      // Open file dialog
      const filePath = await openPythonFile({
        title: 'Open Python File',
      })

      if (!filePath) {
        return false // User cancelled
      }

      // Use the graphStore's API-connected method to parse and load the file
      // This calls the Python sidecar to parse the AST and extract Trinity nodes
      await graphStore.loadFromPythonFile(filePath)

      // Record in recent files
      addRecentFile(filePath)

      return true
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to open file'
      lastError.value = errorMessage
      console.error('Error opening file:', error)

      // Show user-visible error notification
      workspaceStore.addToast({
        severity: 'error',
        summary: 'Failed to open file',
        detail: errorMessage,
        life: 8000,
      })

      return false
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Saves the current file. If no path, prompts for save as.
   */
  async function saveFile(): Promise<boolean> {
    lastError.value = null

    if (!graphStore.currentFilePath) {
      return saveFileAs()
    }

    try {
      isLoading.value = true

      // Get the current graph state
      const graphState = graphStore.getGraphState()

      // Convert to Python content (placeholder for now)
      const content = graphStateToPython(graphState)

      // Write the file
      await writePythonFile(graphStore.currentFilePath, content)

      // Mark as saved and update mtime for file watcher
      graphStore.markSaved()
      await graphStore.updateLastMtime()

      return true
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to save file'
      lastError.value = errorMessage
      console.error('Error saving file:', error)

      workspaceStore.addToast({
        severity: 'error',
        summary: 'Failed to save file',
        detail: errorMessage,
        life: 8000,
      })

      return false
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Saves the file with a new name via native dialog.
   */
  async function saveFileAs(): Promise<boolean> {
    lastError.value = null

    try {
      isLoading.value = true

      // Get default path from current file
      const defaultPath = graphStore.currentFilePath || undefined

      // Open save dialog
      const filePath = await savePythonFileDialog({
        title: 'Save Python File As',
        defaultPath,
      })

      if (!filePath) {
        return false // User cancelled
      }

      // Get the current graph state
      const graphState = graphStore.getGraphState()

      // Convert to Python content (placeholder for now)
      const content = graphStateToPython(graphState)

      // Write the file
      await writePythonFile(filePath, content)

      // Update current file and mark as saved
      graphStore.setCurrentFile(filePath)

      // Record in recent files
      addRecentFile(filePath)

      // Update mtime for file watcher (setCurrentFile triggers watcher start)
      await graphStore.updateLastMtime()

      return true
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to save file'
      lastError.value = errorMessage
      console.error('Error saving file:', error)

      workspaceStore.addToast({
        severity: 'error',
        summary: 'Failed to save file',
        detail: errorMessage,
        life: 8000,
      })

      return false
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Checks if the current file has unsaved changes.
   */
  function hasUnsavedChanges(): boolean {
    return graphStore.isModified
  }

  /**
   * Gets the current file name for display.
   */
  function getFileName(): string {
    return graphStore.fileName
  }

  /**
   * Gets the current file path.
   */
  function getFilePath(): string | null {
    return graphStore.currentFilePath
  }

  /**
   * Generates the current Python code from the graph state.
   * Useful for diff preview before saving.
   */
  function generateCurrentCode(): string {
    const graphState = graphStore.getGraphState()
    return graphStateToPython(graphState)
  }

  return {
    // State
    isLoading,
    lastError,

    // Operations
    newFile,
    openFile,
    saveFile,
    saveFileAs,

    // Helpers
    hasUnsavedChanges,
    getFileName,
    getFilePath,
    confirmUnsavedChanges,
    generateCurrentCode,
  }
}

// =============================================================================
// PYTHON CODE GENERATION
// =============================================================================

/**
 * Python type mapping from common types to Python types.
 */
const TYPE_MAP: Record<string, string> = {
  'string': 'str',
  'number': 'float',
  'integer': 'int',
  'boolean': 'bool',
  'float': 'float',
  'int': 'int',
  'str': 'str',
  'bool': 'bool',
  'any': 'Any',
  'object': 'dict',
  'array': 'list',
  'list': 'list',
  'dict': 'dict',
  'vec2': 'tuple[float, float]',
  'vec3': 'tuple[float, float, float]',
  'color': 'tuple[float, float, float, float]',
}

/**
 * Convert a type string to Python type annotation.
 */
function toPythonType(type: string): string {
  const lowerType = type.toLowerCase()
  return TYPE_MAP[lowerType] || type
}

/**
 * Generate a Python field definition with type annotation.
 */
function generateField(name: string, type: string, defaultValue?: unknown): string {
  const pyType = toPythonType(type)

  if (defaultValue !== undefined) {
    const pyValue = toPythonValue(defaultValue)
    return `    ${name}: ${pyType} = ${pyValue}`
  }

  return `    ${name}: ${pyType}`
}

/**
 * Convert a JavaScript value to Python literal.
 */
function toPythonValue(value: unknown): string {
  if (value === null || value === undefined) {
    return 'None'
  }
  if (typeof value === 'boolean') {
    return value ? 'True' : 'False'
  }
  if (typeof value === 'string') {
    return `"${value.replace(/"/g, '\\"')}"`
  }
  if (typeof value === 'number') {
    return String(value)
  }
  if (Array.isArray(value)) {
    return `[${value.map(toPythonValue).join(', ')}]`
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value)
      .map(([k, v]) => `"${k}": ${toPythonValue(v)}`)
      .join(', ')
    return `{${entries}}`
  }
  return String(value)
}

/**
 * Extract Trinity-specific properties from node properties.
 */
interface TrinityNodeData {
  className: string
  trinityType: string
  fields?: Array<{ name: string; type: string; defaultValue?: unknown }>
  methods?: Array<{ name: string; parameters?: Array<{ name: string; type: string }>; returnType?: string }>
  queries?: string[]
  payloadFields?: Array<{ name: string; type: string }>
  isSingleton?: boolean
  sourceFile?: string
  sourceLine?: number
}

function extractTrinityData(node: GraphNode): TrinityNodeData | null {
  const props = node.properties || {}
  const trinityType = props['trinityType'] as string || node.type.replace('trinity/', '')

  if (!['component', 'system', 'resource', 'event'].includes(trinityType)) {
    return null
  }

  return {
    className: (props['className'] as string) || node.title,
    trinityType,
    fields: props['fields'] as TrinityNodeData['fields'],
    methods: props['methods'] as TrinityNodeData['methods'],
    queries: props['queries'] as string[],
    payloadFields: props['payloadFields'] as TrinityNodeData['payloadFields'],
    isSingleton: props['isSingleton'] as boolean,
    sourceFile: props['sourceFile'] as string,
    sourceLine: props['sourceLine'] as number,
  }
}

/**
 * Generate Python code for a Component node.
 */
function generateComponent(data: TrinityNodeData): string {
  const lines: string[] = [
    `@component`,
    `class ${data.className}:`,
  ]

  if (data.fields && data.fields.length > 0) {
    for (const field of data.fields) {
      lines.push(generateField(field.name, field.type, field.defaultValue))
    }
  } else {
    lines.push('    pass')
  }

  return lines.join('\n')
}

/**
 * Generate Python code for a System node.
 */
function generateSystem(data: TrinityNodeData): string {
  const lines: string[] = [
    `@system`,
    `class ${data.className}:`,
  ]

  // Add fields if any
  if (data.fields && data.fields.length > 0) {
    for (const field of data.fields) {
      lines.push(generateField(field.name, field.type, field.defaultValue))
    }
    lines.push('')
  }

  // Add methods
  if (data.methods && data.methods.length > 0) {
    for (const method of data.methods) {
      const params = ['self']
      if (method.parameters) {
        for (const param of method.parameters) {
          params.push(`${param.name}: ${toPythonType(param.type)}`)
        }
      }

      const returnType = method.returnType ? ` -> ${toPythonType(method.returnType)}` : ''
      lines.push(`    def ${method.name}(${params.join(', ')})${returnType}:`)
      lines.push(`        pass`)
      lines.push('')
    }
  } else if (!data.fields || data.fields.length === 0) {
    lines.push('    pass')
  }

  return lines.join('\n')
}

/**
 * Generate Python code for a Resource node.
 */
function generateResource(data: TrinityNodeData): string {
  const lines: string[] = [
    `@resource`,
    `class ${data.className}:`,
  ]

  if (data.fields && data.fields.length > 0) {
    for (const field of data.fields) {
      lines.push(generateField(field.name, field.type, field.defaultValue))
    }
  } else {
    lines.push('    pass')
  }

  return lines.join('\n')
}

/**
 * Generate Python code for an Event node.
 */
function generateEvent(data: TrinityNodeData): string {
  const lines: string[] = [
    `@event`,
    `class ${data.className}:`,
  ]

  // Use payloadFields for events
  const fields = data.payloadFields || data.fields
  if (fields && fields.length > 0) {
    for (const field of fields) {
      lines.push(generateField(field.name, field.type))
    }
  } else {
    lines.push('    pass')
  }

  return lines.join('\n')
}

/**
 * Generate Python code for a Trinity node based on its type.
 */
function generateNodeCode(data: TrinityNodeData): string {
  switch (data.trinityType) {
    case 'component':
      return generateComponent(data)
    case 'system':
      return generateSystem(data)
    case 'resource':
      return generateResource(data)
    case 'event':
      return generateEvent(data)
    default:
      return `# Unknown node type: ${data.trinityType}`
  }
}

/**
 * Converts a graph state to Python code.
 * Generates valid Python with Trinity ECS decorators.
 */
function graphStateToPython(graphState: GraphState): string {
  const lines: string[] = [
    '"""',
    'FlowForge Generated Python File',
    '',
    'This file was generated from a visual graph.',
    'Edit with caution - changes may be overwritten.',
    '"""',
    '',
    'from trinity import component, system, resource, event',
    'from typing import Any, Optional',
    '',
  ]

  // Group nodes by type for organized output
  const components: TrinityNodeData[] = []
  const systems: TrinityNodeData[] = []
  const resources: TrinityNodeData[] = []
  const events: TrinityNodeData[] = []
  const unknownNodes: GraphNode[] = []

  for (const node of graphState.nodes) {
    const data = extractTrinityData(node)
    if (!data) {
      unknownNodes.push(node)
      continue
    }

    switch (data.trinityType) {
      case 'component':
        components.push(data)
        break
      case 'system':
        systems.push(data)
        break
      case 'resource':
        resources.push(data)
        break
      case 'event':
        events.push(data)
        break
    }
  }

  // Generate Components section
  if (components.length > 0) {
    lines.push('# =============================================================================')
    lines.push('# COMPONENTS')
    lines.push('# =============================================================================')
    lines.push('')
    for (const comp of components) {
      lines.push(generateNodeCode(comp))
      lines.push('')
    }
  }

  // Generate Resources section
  if (resources.length > 0) {
    lines.push('# =============================================================================')
    lines.push('# RESOURCES')
    lines.push('# =============================================================================')
    lines.push('')
    for (const res of resources) {
      lines.push(generateNodeCode(res))
      lines.push('')
    }
  }

  // Generate Events section
  if (events.length > 0) {
    lines.push('# =============================================================================')
    lines.push('# EVENTS')
    lines.push('# =============================================================================')
    lines.push('')
    for (const evt of events) {
      lines.push(generateNodeCode(evt))
      lines.push('')
    }
  }

  // Generate Systems section
  if (systems.length > 0) {
    lines.push('# =============================================================================')
    lines.push('# SYSTEMS')
    lines.push('# =============================================================================')
    lines.push('')
    for (const sys of systems) {
      lines.push(generateNodeCode(sys))
      lines.push('')
    }
  }

  // Add comments for unknown nodes
  if (unknownNodes.length > 0) {
    lines.push('# =============================================================================')
    lines.push('# UNRECOGNIZED NODES (not converted)')
    lines.push('# =============================================================================')
    lines.push('')
    for (const node of unknownNodes) {
      lines.push(`# Node: ${node.id} - Type: ${node.type} - Title: ${node.title}`)
    }
    lines.push('')
  }

  // Add link information as comments
  if (graphState.links.length > 0) {
    lines.push('# =============================================================================')
    lines.push('# GRAPH CONNECTIONS (for reference)')
    lines.push('# =============================================================================')
    lines.push('')
    for (const link of graphState.links) {
      lines.push(`# ${link.originId}[${link.originSlot}] -> ${link.targetId}[${link.targetSlot}] (${link.type})`)
    }
    lines.push('')
  }

  return lines.join('\n')
}

export type UseFileOperationsReturn = ReturnType<typeof useFileOperations>
