/**
 * Node Editing Composable
 *
 * Provides operations for Trinity nodes including:
 * - Node creation and deletion
 * - Field operations (add, remove, update)
 * - Validation and unique name generation
 * - Pending changes tracking
 *
 * @module composables/useNodeEditing
 */
import { ref, computed, type Ref, type ComputedRef } from 'vue'
import { useGraphStore } from '../stores/graphStore'
import type { GraphNode, GraphLink } from '../stores/graphStore'
import { NODE_CONFIG, TRINITY_COLORS, type TrinityColorKey } from '@/config/flowforge.config'
import { PYTHON_KEYWORDS, isPythonKeyword, isValidPythonIdentifier as isValidPythonId } from '@/constants/python'

// =============================================================================
// TYPES
// =============================================================================

/** Trinity node type */
export type TrinityNodeType = 'component' | 'system' | 'resource' | 'event'

/**
 * Field definition for Trinity component nodes.
 */
export interface NodeField {
  /** Field name (must be valid Python identifier) */
  name: string
  /** Field type (e.g., 'float', 'int', 'str', 'Vec3') */
  type: string
  /** Optional default value */
  default?: string
}

/**
 * Data for creating a new node.
 */
export interface NewNodeData {
  type: TrinityNodeType
  className: string
  fields?: NodeField[]
  position?: [number, number]
}

/**
 * Pending change entry for tracking modifications.
 */
export interface PendingChange {
  id: string
  changeType: 'add' | 'delete' | 'modify'
  nodeId?: string
  nodeName?: string
  timestamp: number
}

/**
 * Updates that can be applied to a field.
 */
export interface FieldUpdates {
  name?: string
  type?: string
  default?: string
}

/**
 * Event types emitted by node editing operations.
 */
export type NodeEditEvent =
  | { type: 'field-added'; nodeId: string; field: NodeField }
  | { type: 'field-removed'; nodeId: string; fieldName: string }
  | { type: 'field-updated'; nodeId: string; fieldName: string; updates: FieldUpdates }
  | { type: 'node-modified'; nodeId: string }

/**
 * Options for the useNodeEditing composable.
 */
export interface UseNodeEditingOptions {
  /** Callback when nodes are modified */
  onModified?: (event: NodeEditEvent) => void
  /** Callback before adding a node */
  onBeforeAddNode?: (data: NewNodeData) => boolean | Promise<boolean>
  /** Callback after adding a node */
  onAfterAddNode?: (node: GraphNode) => void
  /** Callback before deleting a node */
  onBeforeDeleteNode?: (nodeId: string) => boolean | Promise<boolean>
  /** Callback after deleting a node */
  onAfterDeleteNode?: (nodeId: string) => void
}

/**
 * Return type for useNodeEditing composable.
 */
export interface UseNodeEditingReturn {
  isModified: ComputedRef<boolean>
  markAsModified: () => void
  clearModified: () => void
  addFieldToNode: (nodeId: string, field: NodeField) => boolean
  removeFieldFromNode: (nodeId: string, fieldName: string) => boolean
  updateNodeField: (nodeId: string, fieldName: string, updates: FieldUpdates) => boolean
  getNodeFields: (nodeId: string) => NodeField[]
  fieldExists: (nodeId: string, fieldName: string) => boolean
  isValidPythonIdentifier: (name: string) => boolean
  validateFieldName: (nodeId: string, name: string, excludeCurrent?: string) => string | null
  addNewNode: (data: NewNodeData) => Promise<GraphNode | null>
  deleteNode: (nodeId: string) => Promise<boolean>
  deleteNodes: (nodeIds: string[]) => Promise<boolean>
  deleteSelectedNodes: () => Promise<boolean>
  getConnectedEdges: (nodeId: string) => GraphLink[]
  classNameExists: (className: string, excludeNodeId?: string) => boolean
  generateUniqueClassName: (baseName: string, type: TrinityNodeType) => string
  getDefaultClassName: (type: TrinityNodeType) => string
  validateClassName: (name: string) => string | null
  getDecoratorName: (type: TrinityNodeType) => string
  getLiteGraphType: (type: TrinityNodeType) => string
  getNodeColor: (type: TrinityNodeType) => (typeof TRINITY_COLORS)[TrinityColorKey]
  pendingChanges: Ref<PendingChange[]>
  pendingChangeCount: ComputedRef<number>
  hasPendingChanges: ComputedRef<boolean>
  clearPendingChanges: () => void
  getPendingChangesSummary: () => string
  isProcessing: Ref<boolean>
  lastError: Ref<string | null>
}

// =============================================================================
// PYTHON IDENTIFIER VALIDATION
// =============================================================================

const PYTHON_IDENTIFIER_REGEX = /^[a-zA-Z_][a-zA-Z0-9_]*$/

// Use imported isValidPythonIdentifier from @/constants/python (aliased as isValidPythonId)
function isValidPythonIdentifier(name: string): boolean {
  return isValidPythonId(name)
}

// =============================================================================
// COMPOSABLE
// =============================================================================

export function useNodeEditing(options: UseNodeEditingOptions = {}): UseNodeEditingReturn {
  const graphStore = useGraphStore()
  const localModified = ref(false)
  const isModified = computed(() => localModified.value || graphStore.isModified)

  function markAsModified(): void {
    localModified.value = true
    graphStore.markModified()
  }

  function clearModified(): void {
    localModified.value = false
    graphStore.markSaved()
  }

  function emitEvent(event: NodeEditEvent): void {
    options.onModified?.(event)
  }

  function getNode(nodeId: string): GraphNode | undefined {
    return graphStore.nodes.find(n => n.id === nodeId)
  }

  function getNodeFields(nodeId: string): NodeField[] {
    const node = getNode(nodeId)
    if (!node) return []
    const props = node.properties ?? {}
    const fields = props['fields'] as NodeField[] | undefined
    if (fields && Array.isArray(fields)) return fields
    const componentData = props['componentData'] as { fields?: NodeField[] } | undefined
    if (componentData?.fields && Array.isArray(componentData.fields)) return componentData.fields
    return []
  }

  function fieldExists(nodeId: string, fieldName: string): boolean {
    const fields = getNodeFields(nodeId)
    return fields.some(f => f.name === fieldName)
  }

  function validateFieldName(nodeId: string, name: string, excludeCurrent?: string): string | null {
    if (!name || name.trim().length === 0) return 'Field name is required'
    const trimmedName = name.trim()
    if (!isValidPythonIdentifier(trimmedName)) {
      if (PYTHON_KEYWORDS.has(trimmedName)) return `'${trimmedName}' is a Python reserved keyword`
      return 'Must be a valid Python identifier'
    }
    const fields = getNodeFields(nodeId)
    const duplicate = fields.find(f => f.name === trimmedName && f.name !== excludeCurrent)
    if (duplicate) return `Field '${trimmedName}' already exists`
    return null
  }

  function addFieldToNode(nodeId: string, field: NodeField): boolean {
    const node = getNode(nodeId)
    if (!node) return false
    const validationError = validateFieldName(nodeId, field.name)
    if (validationError) return false
    const props = { ...(node.properties ?? {}) }
    if (props['componentData']) {
      const componentData = { ...(props['componentData'] as { fields?: NodeField[] }) }
      const fields = [...(componentData.fields ?? [])]
      fields.push({ ...field })
      componentData.fields = fields
      props['componentData'] = componentData
    } else {
      const existingFields = (props['fields'] as NodeField[] | undefined) ?? []
      const fields = [...existingFields]
      fields.push({ ...field })
      props['fields'] = fields
    }
    graphStore.updateNode(nodeId, { properties: props })
    markAsModified()
    emitEvent({ type: 'field-added', nodeId, field })
    emitEvent({ type: 'node-modified', nodeId })
    return true
  }

  function removeFieldFromNode(nodeId: string, fieldName: string): boolean {
    const node = getNode(nodeId)
    if (!node) return false
    const props = { ...(node.properties ?? {}) }
    let removed = false
    if (props['componentData']) {
      const componentData = { ...(props['componentData'] as { fields?: NodeField[] }) }
      const fields = componentData.fields ?? []
      const originalLength = fields.length
      componentData.fields = fields.filter(f => f.name !== fieldName)
      removed = componentData.fields.length < originalLength
      props['componentData'] = componentData
    } else if (props['fields']) {
      const fields = props['fields'] as NodeField[]
      const originalLength = fields.length
      props['fields'] = fields.filter(f => f.name !== fieldName)
      removed = (props['fields'] as NodeField[]).length < originalLength
    }
    if (!removed) return false
    graphStore.updateNode(nodeId, { properties: props })
    markAsModified()
    emitEvent({ type: 'field-removed', nodeId, fieldName })
    emitEvent({ type: 'node-modified', nodeId })
    return true
  }

  function updateNodeField(nodeId: string, fieldName: string, updates: FieldUpdates): boolean {
    const node = getNode(nodeId)
    if (!node) return false
    if (updates.name !== undefined && updates.name !== fieldName) {
      const validationError = validateFieldName(nodeId, updates.name, fieldName)
      if (validationError) return false
    }
    const props = { ...(node.properties ?? {}) }
    let updated = false
    const updateField = (fields: NodeField[]): NodeField[] => {
      return fields.map(f => {
        if (f.name === fieldName) {
          updated = true
          const result: NodeField = { name: updates.name ?? f.name, type: updates.type ?? f.type }
          const newDefault = updates.default !== undefined ? updates.default : f.default
          if (newDefault !== undefined) result.default = newDefault
          return result
        }
        return f
      })
    }
    if (props['componentData']) {
      const componentData = { ...(props['componentData'] as { fields?: NodeField[] }) }
      if (componentData.fields) {
        componentData.fields = updateField(componentData.fields)
        props['componentData'] = componentData
      }
    } else if (props['fields']) {
      props['fields'] = updateField(props['fields'] as NodeField[])
    }
    if (!updated) return false
    graphStore.updateNode(nodeId, { properties: props })
    markAsModified()
    emitEvent({ type: 'field-updated', nodeId, fieldName, updates })
    emitEvent({ type: 'node-modified', nodeId })
    return true
  }

  // Node operations
  const pendingChanges = ref<PendingChange[]>([])
  const isProcessing = ref(false)
  const lastError = ref<string | null>(null)

  function validateClassName(name: string): string | null {
    if (!name || name.trim().length === 0) return 'Class name is required'
    const trimmed = name.trim()
    if (!PYTHON_IDENTIFIER_REGEX.test(trimmed)) {
      if (/^[0-9]/.test(trimmed)) return 'Class name cannot start with a number'
      if (/\s/.test(trimmed)) return 'Class name cannot contain spaces'
      return 'Class name can only contain letters, numbers, and underscores'
    }
    if (PYTHON_KEYWORDS.has(trimmed)) return `"${trimmed}" is a Python reserved keyword`
    return null
  }

  function getDecoratorName(type: TrinityNodeType): string {
    return `@${type}`
  }

  function getLiteGraphType(type: TrinityNodeType): string {
    return `trinity/${type}`
  }

  function getNodeColor(type: TrinityNodeType): (typeof TRINITY_COLORS)[TrinityColorKey] {
    return TRINITY_COLORS[type as TrinityColorKey]
  }

  function generateUniqueClassName(baseName: string, type: TrinityNodeType): string {
    const existingNames = new Set(
      graphStore.nodes
        .filter(n => n.type === getLiteGraphType(type) || n.type === type)
        .map(n => (n.properties?.['className'] as string) || n.title)
    )
    if (!existingNames.has(baseName)) return baseName
    let counter = 1
    let candidate = `${baseName}${counter}`
    while (existingNames.has(candidate)) {
      counter++
      candidate = `${baseName}${counter}`
    }
    return candidate
  }

  function getDefaultClassName(type: TrinityNodeType): string {
    const baseNames: Record<TrinityNodeType, string> = {
      component: 'NewComponent',
      system: 'NewSystem',
      resource: 'NewResource',
      event: 'NewEvent'
    }
    return generateUniqueClassName(baseNames[type], type)
  }

  function classNameExists(className: string, excludeNodeId?: string): boolean {
    return graphStore.nodes.some(node => {
      if (excludeNodeId && node.id === excludeNodeId) return false
      const nodeClassName = (node.properties?.['className'] as string) || node.title
      return nodeClassName === className
    })
  }

  function getDefaultInputs(type: TrinityNodeType): GraphNode['inputs'] {
    switch (type) {
      case 'system': return [{ name: 'trigger', type: 'exec', link: null }]
      case 'event': return [{ name: 'trigger', type: 'exec', link: null }]
      default: return []
    }
  }

  function getDefaultOutputs(type: TrinityNodeType): GraphNode['outputs'] {
    switch (type) {
      case 'component': return [{ name: 'data', type: 'component', links: [] }]
      case 'system': return [{ name: 'next', type: 'exec', links: [] }]
      case 'resource': return [{ name: 'data', type: 'resource', links: [] }]
      case 'event': return [{ name: 'signal', type: 'event', links: [] }]
      default: return []
    }
  }

  async function addNewNode(data: NewNodeData): Promise<GraphNode | null> {
    lastError.value = null
    const validationError = validateClassName(data.className)
    if (validationError) {
      lastError.value = validationError
      return null
    }
    if (options.onBeforeAddNode) {
      const shouldProceed = await options.onBeforeAddNode(data)
      if (!shouldProceed) return null
    }
    try {
      isProcessing.value = true
      const position = data.position ?? [100, 100]
      const size = NODE_CONFIG.sizes[data.type] || NODE_CONFIG.defaultSize
      const nodeData = {
        type: getLiteGraphType(data.type),
        title: data.className,
        pos: position as [number, number],
        size: [...size] as [number, number],
        properties: {
          trinityType: data.type,
          className: data.className,
          fields: data.fields ?? [],
          sourceFile: graphStore.currentFilePath || undefined
        },
        inputs: getDefaultInputs(data.type) || [],
        outputs: getDefaultOutputs(data.type) || [],
        flags: {}
      } as Omit<GraphNode, 'id'>
      const newNode = graphStore.addNode(nodeData)
      pendingChanges.value.push({
        id: `add_${newNode.id}_${Date.now()}`,
        changeType: 'add',
        nodeId: newNode.id,
        nodeName: data.className,
        timestamp: Date.now()
      })
      if (options.onAfterAddNode) options.onAfterAddNode(newNode)
      return newNode
    } catch (error) {
      lastError.value = error instanceof Error ? error.message : 'Failed to add node'
      return null
    } finally {
      isProcessing.value = false
    }
  }

  async function deleteNode(nodeId: string): Promise<boolean> {
    lastError.value = null
    const node = getNode(nodeId)
    if (!node) {
      lastError.value = 'Node not found'
      return false
    }
    if (options.onBeforeDeleteNode) {
      const shouldProceed = await options.onBeforeDeleteNode(nodeId)
      if (!shouldProceed) return false
    }
    try {
      isProcessing.value = true
      const nodeName = (node.properties?.['className'] as string) || node.title
      graphStore.removeNode(nodeId)
      pendingChanges.value.push({
        id: `delete_${nodeId}_${Date.now()}`,
        changeType: 'delete',
        nodeId,
        nodeName,
        timestamp: Date.now()
      })
      if (options.onAfterDeleteNode) options.onAfterDeleteNode(nodeId)
      return true
    } catch (error) {
      lastError.value = error instanceof Error ? error.message : 'Failed to delete node'
      return false
    } finally {
      isProcessing.value = false
    }
  }

  async function deleteNodes(nodeIds: string[]): Promise<boolean> {
    if (nodeIds.length === 0) return true
    let success = true
    for (const nodeId of nodeIds) {
      const result = await deleteNode(nodeId)
      if (!result) success = false
    }
    return success
  }

  async function deleteSelectedNodes(): Promise<boolean> {
    const selectedIds = Array.from(graphStore.selectedNodeIds)
    if (selectedIds.length === 0) return true
    return deleteNodes(selectedIds)
  }

  function getConnectedEdges(nodeId: string) {
    return graphStore.links.filter(link => link.originId === nodeId || link.targetId === nodeId)
  }

  const pendingChangeCount = computed(() => pendingChanges.value.length)
  const hasPendingChanges = computed(() => pendingChanges.value.length > 0)

  function clearPendingChanges(): void {
    pendingChanges.value = []
  }

  function getPendingChangesSummary(): string {
    const adds = pendingChanges.value.filter(c => c.changeType === 'add').length
    const deletes = pendingChanges.value.filter(c => c.changeType === 'delete').length
    const modifies = pendingChanges.value.filter(c => c.changeType === 'modify').length
    const parts: string[] = []
    if (adds > 0) parts.push(`${adds} added`)
    if (deletes > 0) parts.push(`${deletes} deleted`)
    if (modifies > 0) parts.push(`${modifies} modified`)
    return parts.join(', ') || 'No changes'
  }

  return {
    isModified,
    markAsModified,
    clearModified,
    addFieldToNode,
    removeFieldFromNode,
    updateNodeField,
    getNodeFields,
    fieldExists,
    isValidPythonIdentifier,
    validateFieldName,
    addNewNode,
    deleteNode,
    deleteNodes,
    deleteSelectedNodes,
    getConnectedEdges,
    classNameExists,
    generateUniqueClassName,
    getDefaultClassName,
    validateClassName,
    getDecoratorName,
    getLiteGraphType,
    getNodeColor,
    pendingChanges,
    pendingChangeCount,
    hasPendingChanges,
    clearPendingChanges,
    getPendingChangesSummary,
    isProcessing,
    lastError
  }
}

export default useNodeEditing
