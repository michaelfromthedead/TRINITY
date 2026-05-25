/**
 * Inspector Composable
 *
 * Provides integration between graph selection and the Inspector panel.
 * Fetches and formats inspector data for selected Trinity nodes including
 * class hierarchy, decorators, metaclass info, fields, and methods.
 *
 * Uses Trinity Foundation APIs via the bridge for detailed inspection data.
 *
 * @module composables/useInspector
 */

import { ref, computed, watch, readonly, type Ref, type ComputedRef, shallowRef } from 'vue'
import { useGraphStore } from '@/stores/graphStore'
import { inspectType, type InspectionResult } from '@/bridge/trinity'
import { openInEditor } from '@/bridge/editor'
import type {
  InspectorData,
  InspectorSectionState,
  DecoratorInfo,
  DecoratorCategory,
  MetaclassInfo,
  InheritanceChain,
  BaseClassInfo,
  FieldInfo,
  MethodInfo,
  MethodParam,
  TrinityType,
  SourceInfo,
  DecoratorArg,
} from '@/types/inspector'
import { DEFAULT_SECTION_STATE } from '@/types/inspector'

// =============================================================================
// TYPES
// =============================================================================

/**
 * Options for the useInspector composable.
 */
export interface UseInspectorOptions {
  /** Whether to automatically fetch data when selection changes */
  autoFetch?: boolean

  /** Callback when inspector data is loaded */
  onDataLoaded?: (data: InspectorData) => void

  /** Callback when an error occurs */
  onError?: (error: Error) => void
}

/**
 * Return type for the useInspector composable.
 */
export interface UseInspectorReturn {
  /** Current inspector data (null if nothing selected) */
  readonly inspectorData: ComputedRef<InspectorData | null>

  /** Whether data is currently being fetched */
  readonly isLoading: Ref<boolean>

  /** Error message if fetch failed */
  readonly error: Ref<string | null>

  /** Whether a node is currently selected */
  readonly hasSelection: ComputedRef<boolean>

  /** Section collapsed states */
  readonly sectionState: Ref<InspectorSectionState>

  /** Fetch inspector data for a specific node */
  fetchInspectorData: (nodeId: string) => Promise<void>

  /** Clear the current inspector data */
  clearInspectorData: () => void

  /** Toggle a section's collapsed state */
  toggleSection: (section: keyof InspectorSectionState) => void

  /** Format a decorator for display */
  formatDecorator: (decorator: DecoratorInfo) => string

  /** Format a method signature for display */
  formatMethodSignature: (method: MethodInfo) => string

  /** Get the color for a decorator category */
  getDecoratorColor: (category: DecoratorCategory) => string

  /** Get the color for a Trinity type */
  getTrinityTypeColor: (type: TrinityType) => string

  /** Open source file in editor */
  openInEditor: (source: SourceInfo) => void
}

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Color mapping for decorator categories.
 */
const DECORATOR_COLORS: Record<DecoratorCategory, string> = {
  component: 'var(--flowforge-node-component, #3b82f6)',
  system: 'var(--flowforge-node-system, #22c55e)',
  resource: 'var(--flowforge-node-resource, #a855f7)',
  event: 'var(--flowforge-node-event, #f97316)',
  builtin: 'var(--flowforge-text-muted, #666680)',
  custom: 'var(--flowforge-text, #e0e0e0)',
}

/**
 * Color mapping for Trinity types.
 */
const TRINITY_TYPE_COLORS: Record<TrinityType, string> = {
  component: 'var(--flowforge-node-component, #3b82f6)',
  system: 'var(--flowforge-node-system, #22c55e)',
  resource: 'var(--flowforge-node-resource, #a855f7)',
  event: 'var(--flowforge-node-event, #f97316)',
  unknown: 'var(--flowforge-text-muted, #666680)',
}

/**
 * Known Trinity decorator names and their categories.
 */
const TRINITY_DECORATORS: Record<string, DecoratorCategory> = {
  component: 'component',
  system: 'system',
  resource: 'resource',
  event: 'event',
}

/**
 * Known Python built-in decorators.
 */
const BUILTIN_DECORATORS = new Set([
  'staticmethod',
  'classmethod',
  'property',
  'abstractmethod',
  'dataclass',
  'cached_property',
  'functools.lru_cache',
  'functools.cached_property',
])

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Safely get a property from an object with index access.
 */
function getProp<T>(obj: Record<string, unknown>, key: string): T | undefined {
  return obj[key] as T | undefined
}

// =============================================================================
// COMPOSABLE
// =============================================================================

/**
 * Composable for managing Inspector panel state and data.
 *
 * @example
 * ```typescript
 * const {
 *   inspectorData,
 *   isLoading,
 *   hasSelection,
 *   sectionState,
 *   toggleSection,
 *   formatDecorator,
 * } = useInspector({
 *   autoFetch: true,
 *   onDataLoaded: (data) => console.log('Loaded:', data.className),
 * })
 * ```
 */
export function useInspector(options: UseInspectorOptions = {}): UseInspectorReturn {
  const { autoFetch = true, onDataLoaded, onError } = options

  // Get graph store for selection state
  const graphStore = useGraphStore()

  // ---------------------------------------------------------------------------
  // STATE
  // ---------------------------------------------------------------------------

  const inspectorDataRef = shallowRef<InspectorData | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const sectionState = ref<InspectorSectionState>({ ...DEFAULT_SECTION_STATE })

  // ---------------------------------------------------------------------------
  // COMPUTED
  // ---------------------------------------------------------------------------

  const inspectorData = computed(() => inspectorDataRef.value)

  const hasSelection = computed(() => {
    return graphStore.selectedNodeIds.size === 1
  })

  // ---------------------------------------------------------------------------
  // WATCHERS
  // ---------------------------------------------------------------------------

  // Watch for selection changes and auto-fetch data
  if (autoFetch) {
    watch(
      () => Array.from(graphStore.selectedNodeIds),
      async (selectedIds) => {
        if (selectedIds.length === 1 && selectedIds[0]) {
          await fetchInspectorData(selectedIds[0])
        } else {
          clearInspectorData()
        }
      },
      { immediate: true }
    )
  }

  // ---------------------------------------------------------------------------
  // METHODS
  // ---------------------------------------------------------------------------

  /**
   * Fetch inspector data for a specific node.
   * First tries to get data from Trinity bridge, falls back to node properties.
   */
  async function fetchInspectorData(nodeId: string): Promise<void> {
    isLoading.value = true
    error.value = null

    try {
      // Get the node from the graph store
      const node = graphStore.nodes.find((n) => n.id === nodeId)

      if (!node) {
        throw new Error(`Node not found: ${nodeId}`)
      }

      // Try to get detailed data from Trinity bridge first
      const typeName = node.title || (node.properties?.['class_name'] as string)
      let trinityData: InspectionResult | null = null

      if (typeName) {
        try {
          trinityData = await inspectType(typeName)
          if (!trinityData.success) {
            console.log('[useInspector] Trinity inspection failed, using node properties:', trinityData.error)
            trinityData = null
          }
        } catch (bridgeErr) {
          console.log('[useInspector] Trinity bridge unavailable, using node properties')
        }
      }

      // Build inspector data, merging Trinity data with node properties
      const data = buildInspectorDataFromNode(node, nodeId, trinityData)

      inspectorDataRef.value = data

      if (onDataLoaded) {
        onDataLoaded(data)
      }

      console.log('[useInspector] Loaded data for:', data.className)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch inspector data'
      error.value = errorMessage
      console.error('[useInspector] Error:', errorMessage)

      if (onError && err instanceof Error) {
        onError(err)
      }
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Build inspector data from a graph node.
   * Optionally merges with Trinity inspection data if available.
   */
  function buildInspectorDataFromNode(
    node: { id: string; type: string; title: string; properties?: Record<string, unknown> },
    nodeId: string,
    trinityData?: InspectionResult | null
  ): InspectorData {
    const props = node.properties || {}

    // Extract source info - prefer Trinity data if available
    let sourceFile = getProp<string>(props, 'source_file') || getProp<string>(props, 'file') || ''
    let sourceLine = getProp<number>(props, 'source_line') || getProp<number>(props, 'line') || 0
    const endLine = getProp<number>(props, 'end_line')

    if (trinityData?.source) {
      sourceFile = trinityData.source.file || sourceFile
      sourceLine = trinityData.source.line ?? sourceLine
    }

    const source: SourceInfo = {
      file: sourceFile,
      line: sourceLine,
    }
    if (endLine !== undefined) {
      source.endLine = endLine
    }

    // Determine Trinity type - prefer Trinity data
    let trinityType = determineTrinityType(node.type)
    if (trinityData?.category) {
      const category = trinityData.category.toLowerCase()
      if (['component', 'system', 'resource', 'event'].includes(category)) {
        trinityType = category as TrinityType
      }
    }

    // Build decorators list - merge Trinity data
    const decorators = buildDecorators(props, trinityType, trinityData)

    // Build inheritance chain - merge Trinity data
    const inheritance = buildInheritanceChain(props, trinityData)

    // Build metaclass info - prefer Trinity data
    const metaclass = buildMetaclassInfo(props, trinityData)

    // Build fields list - merge Trinity data
    const fields = buildFields(props, trinityData)

    // Build methods list
    const methods = buildMethods(props)

    // Extract class and module name - prefer Trinity data
    const className = trinityData?.name || node.title || getProp<string>(props, 'class_name') || 'Unknown'
    const moduleName = trinityData?.module || extractModuleName(sourceFile, props)
    const qualifiedName = trinityData?.qualifiedName || (moduleName ? `${moduleName}.${className}` : className)

    const result: InspectorData = {
      nodeId,
      className,
      moduleName,
      qualifiedName,
      trinityType,
      source,
      inheritance,
      decorators,
      fields,
      methods,
    }

    // Add optional properties - prefer Trinity data
    const docstring = trinityData?.doc || getProp<string>(props, 'docstring')
    if (docstring !== undefined) {
      result.docstring = docstring
    }

    if (metaclass !== undefined) {
      result.metaclass = metaclass
    }

    const rawSource = getProp<string>(props, 'raw_source')
    if (rawSource !== undefined) {
      result.rawSource = rawSource
    }

    return result
  }

  /**
   * Determine Trinity type from node type string.
   */
  function determineTrinityType(nodeType: string): TrinityType {
    const lowerType = nodeType.toLowerCase()

    if (lowerType.includes('component')) return 'component'
    if (lowerType.includes('system')) return 'system'
    if (lowerType.includes('resource')) return 'resource'
    if (lowerType.includes('event')) return 'event'

    return 'unknown'
  }

  /**
   * Build decorators list from node properties and Trinity data.
   */
  function buildDecorators(
    props: Record<string, unknown>,
    trinityType: TrinityType,
    trinityData?: InspectionResult | null
  ): DecoratorInfo[] {
    const decorators: DecoratorInfo[] = []
    const addedNames = new Set<string>()

    // Add decorators from Trinity data first (higher quality)
    if (trinityData?.decorators && Array.isArray(trinityData.decorators)) {
      for (const dec of trinityData.decorators) {
        const category = determineCategoryFromDecorator(dec.name, dec.foundation)
        const decoratorInfo: DecoratorInfo = {
          name: dec.name,
          category,
          source: `@${dec.name}`,
        }
        if (dec.args) {
          decoratorInfo.args = parseDecoratorArgsFromObj(dec.args as Record<string, unknown>)
        }
        decorators.push(decoratorInfo)
        addedNames.add(dec.name)
      }
    }

    // Add Trinity type decorator if not already added and not unknown
    if (trinityType !== 'unknown' && !addedNames.has(trinityType)) {
      decorators.unshift({
        name: trinityType,
        category: trinityType as DecoratorCategory,
        source: `@${trinityType}`,
      })
      addedNames.add(trinityType)
    }

    // Parse additional decorators from properties
    const decoratorsList = getProp<string[]>(props, 'decorators')
    if (Array.isArray(decoratorsList)) {
      for (const dec of decoratorsList) {
        const parsed = parseDecoratorString(dec)
        // Avoid duplicates
        if (!addedNames.has(parsed.name)) {
          decorators.push(parsed)
          addedNames.add(parsed.name)
        }
      }
    }

    return decorators
  }

  /**
   * Determine decorator category from name and foundation flag.
   */
  function determineCategoryFromDecorator(name: string, foundation?: boolean): DecoratorCategory {
    const lowerName = name.toLowerCase()
    if (lowerName === 'component') return 'component'
    if (lowerName === 'system') return 'system'
    if (lowerName === 'resource') return 'resource'
    if (lowerName === 'event') return 'event'
    if (foundation) return 'builtin'
    if (BUILTIN_DECORATORS.has(name)) return 'builtin'
    return 'custom'
  }

  /**
   * Parse decorator arguments from an object.
   */
  function parseDecoratorArgsFromObj(argsObj: Record<string, unknown>): DecoratorArg[] {
    return Object.entries(argsObj).map(([key, value]): DecoratorArg => ({
      key,
      value: String(value),
    }))
  }

  /**
   * Parse a decorator string into DecoratorInfo.
   */
  function parseDecoratorString(decoratorStr: string): DecoratorInfo {
    // Remove @ prefix if present
    const str = decoratorStr.startsWith('@') ? decoratorStr.slice(1) : decoratorStr

    // Extract name and args
    const parenIndex = str.indexOf('(')
    let name: string
    let argsStr: string | undefined

    if (parenIndex !== -1) {
      name = str.slice(0, parenIndex).trim()
      argsStr = str.slice(parenIndex + 1, -1).trim()
    } else {
      name = str.trim()
    }

    // Determine category
    let category: DecoratorCategory = 'custom'
    const trinityCategory = TRINITY_DECORATORS[name]
    if (trinityCategory) {
      category = trinityCategory
    } else if (BUILTIN_DECORATORS.has(name)) {
      category = 'builtin'
    }

    const result: DecoratorInfo = {
      name,
      category,
      source: decoratorStr,
    }

    if (argsStr) {
      result.args = parseDecoratorArgs(argsStr)
    }

    return result
  }

  /**
   * Parse decorator arguments string.
   */
  function parseDecoratorArgs(argsStr: string): DecoratorArg[] {
    // Simple parsing - split by comma (doesn't handle nested structures)
    const parts = argsStr.split(',').map((s) => s.trim()).filter(Boolean)

    return parts.map((part, index): DecoratorArg => {
      const eqIndex = part.indexOf('=')
      if (eqIndex !== -1) {
        return {
          key: part.slice(0, eqIndex).trim(),
          value: part.slice(eqIndex + 1).trim(),
        }
      }
      return {
        key: index,
        value: part,
      }
    })
  }

  /**
   * Build inheritance chain from node properties and Trinity data.
   */
  function buildInheritanceChain(
    props: Record<string, unknown>,
    trinityData?: InspectionResult | null
  ): InheritanceChain {
    const bases: BaseClassInfo[] = []
    const addedNames = new Set<string>()

    // Use Trinity hierarchy data if available (higher quality)
    if (trinityData?.hierarchy && Array.isArray(trinityData.hierarchy)) {
      for (const entry of trinityData.hierarchy) {
        const baseInfo: BaseClassInfo = {
          name: entry.name,
          isTrinityBase: entry.isTrinityBase,
        }
        if (entry.module) {
          baseInfo.module = entry.module
        }
        bases.push(baseInfo)
        addedNames.add(entry.name)
      }
    }

    // Parse bases from properties as fallback/supplement
    const basesList = getProp<Array<string | { name: string; module?: string }>>(props, 'bases')

    if (Array.isArray(basesList)) {
      for (const base of basesList) {
        if (typeof base === 'string') {
          if (!addedNames.has(base)) {
            bases.push({
              name: base,
              isTrinityBase: isTrinityBaseClass(base),
            })
            addedNames.add(base)
          }
        } else if (base && typeof base === 'object') {
          if (!addedNames.has(base.name)) {
            const baseInfo: BaseClassInfo = {
              name: base.name,
              isTrinityBase: isTrinityBaseClass(base.name),
            }
            if (base.module) {
              baseInfo.module = base.module
            }
            bases.push(baseInfo)
            addedNames.add(base.name)
          }
        }
      }
    }

    const result: InheritanceChain = {
      bases,
      isMultipleInheritance: bases.length > 1,
    }

    // Parse MRO if available
    const mro = getProp<string[]>(props, 'mro')
    if (mro !== undefined) {
      result.mro = mro
    }

    return result
  }

  /**
   * Check if a class name is a known Trinity base class.
   */
  function isTrinityBaseClass(name: string): boolean {
    const trinityBases = ['Component', 'System', 'Resource', 'Event', 'Entity', 'World']
    return trinityBases.includes(name)
  }

  /**
   * Build metaclass info from node properties and Trinity data.
   */
  function buildMetaclassInfo(
    props: Record<string, unknown>,
    trinityData?: InspectionResult | null
  ): MetaclassInfo | undefined {
    // Prefer Trinity data for metaclass
    const metaclassName = trinityData?.metaclass || getProp<string>(props, 'metaclass')

    if (!metaclassName) {
      return undefined
    }

    const trinityMetas = ['ComponentMeta', 'SystemMeta', 'ResourceMeta', 'EventMeta', 'EntityMeta', 'AssetMeta', 'ProtocolMeta', 'StateMeta']

    const result: MetaclassInfo = {
      name: metaclassName,
      isTrinityMeta: trinityMetas.includes(metaclassName),
    }

    const moduleVal = getProp<string>(props, 'metaclass_module')
    if (moduleVal !== undefined) {
      result.module = moduleVal
    }

    // Build description based on metaclass type
    const descVal = getProp<string>(props, 'metaclass_description')
    if (descVal !== undefined) {
      result.description = descVal
    } else if (result.isTrinityMeta) {
      // Provide default descriptions for Trinity metaclasses
      const descriptions: Record<string, string> = {
        ComponentMeta: 'Registers component types and manages field schemas',
        SystemMeta: 'Registers systems and manages query specifications',
        ResourceMeta: 'Registers resources and manages singleton instances',
        EventMeta: 'Registers event types and manages event channels',
        EntityMeta: 'Manages entity archetypes and component relationships',
        AssetMeta: 'Registers asset types and manages file extensions',
        ProtocolMeta: 'Manages protocol versioning and serialization',
        StateMeta: 'Manages state machine transitions and lifecycle',
      }
      const desc = descriptions[metaclassName]
      if (desc) {
        result.description = desc
      }
    }

    return result
  }

  /**
   * Build fields list from node properties and Trinity data.
   */
  function buildFields(
    props: Record<string, unknown>,
    trinityData?: InspectionResult | null
  ): FieldInfo[] {
    const fields: FieldInfo[] = []
    const addedNames = new Set<string>()

    // Build fields from Trinity data first (higher quality)
    if (trinityData?.fieldTypes) {
      const fieldTypes = trinityData.fieldTypes as Record<string, string>
      const fieldDefaults = (trinityData.fieldDefaults || {}) as Record<string, unknown>

      for (const [name, type] of Object.entries(fieldTypes)) {
        const hasDefault = name in fieldDefaults
        const fieldInfo: FieldInfo = {
          name,
          type: type || 'Any',
          hasDefault,
          isClassVar: false,
        }

        if (hasDefault) {
          fieldInfo.default = String(fieldDefaults[name])
        }

        fields.push(fieldInfo)
        addedNames.add(name)
      }
    }

    // Parse fields from properties as fallback/supplement
    const fieldsList = getProp<Array<{
      name: string
      type?: string
      default?: string
      is_class_var?: boolean
      doc?: string
      line?: number
    }>>(props, 'fields')

    if (Array.isArray(fieldsList)) {
      for (const field of fieldsList) {
        if (addedNames.has(field.name)) {
          // Update existing field with additional info
          const existing = fields.find(f => f.name === field.name)
          if (existing) {
            if (field.doc !== undefined) {
              existing.doc = field.doc
            }
            if (field.line !== undefined) {
              existing.line = field.line
            }
            if (field.is_class_var !== undefined) {
              existing.isClassVar = field.is_class_var
            }
          }
          continue
        }

        const fieldInfo: FieldInfo = {
          name: field.name,
          type: field.type || 'Any',
          hasDefault: field.default !== undefined,
          isClassVar: field.is_class_var || false,
        }

        if (field.default !== undefined) {
          fieldInfo.default = field.default
        }
        if (field.doc !== undefined) {
          fieldInfo.doc = field.doc
        }
        if (field.line !== undefined) {
          fieldInfo.line = field.line
        }

        fields.push(fieldInfo)
        addedNames.add(field.name)
      }
    }

    return fields
  }

  /**
   * Build methods list from node properties.
   */
  function buildMethods(props: Record<string, unknown>): MethodInfo[] {
    const methods: MethodInfo[] = []

    // Parse methods from properties
    const methodsList = getProp<Array<{
      name: string
      signature?: string
      return_type?: string
      params?: Array<{
        name: string
        type?: string
        default?: string
        var_positional?: boolean
        var_keyword?: boolean
      }>
      decorators?: string[]
      is_static?: boolean
      is_classmethod?: boolean
      is_property?: boolean
      is_abstract?: boolean
      doc?: string
      line?: number
    }>>(props, 'methods')

    if (Array.isArray(methodsList)) {
      for (const method of methodsList) {
        const params: MethodParam[] = (method.params || []).map((p): MethodParam => {
          const param: MethodParam = {
            name: p.name,
            isVarPositional: p.var_positional || false,
            isVarKeyword: p.var_keyword || false,
          }
          if (p.type !== undefined) {
            param.type = p.type
          }
          if (p.default !== undefined) {
            param.default = p.default
          }
          return param
        })

        const methodInfo: MethodInfo = {
          name: method.name,
          signature: method.signature || buildMethodSignatureFromData(method),
          params,
          decorators: method.decorators || [],
          isStatic: method.is_static || false,
          isClassMethod: method.is_classmethod || false,
          isProperty: method.is_property || false,
          isAbstract: method.is_abstract || false,
        }

        if (method.return_type !== undefined) {
          methodInfo.returnType = method.return_type
        }
        if (method.doc !== undefined) {
          methodInfo.doc = method.doc
        }
        if (method.line !== undefined) {
          methodInfo.line = method.line
        }

        methods.push(methodInfo)
      }
    }

    return methods
  }

  /**
   * Build a method signature string from method data.
   */
  function buildMethodSignatureFromData(method: {
    name: string
    params?: Array<{
      name: string
      type?: string
      default?: string
      var_positional?: boolean
      var_keyword?: boolean
    }>
    return_type?: string
  }): string {
    const params = method.params || []
    const paramStrings = params.map((p) => {
      let str = p.var_positional ? `*${p.name}` : p.var_keyword ? `**${p.name}` : p.name
      if (p.type) {
        str += `: ${p.type}`
      }
      if (p.default !== undefined) {
        str += ` = ${p.default}`
      }
      return str
    })

    const returnType = method.return_type ? ` -> ${method.return_type}` : ''

    return `def ${method.name}(${paramStrings.join(', ')})${returnType}`
  }

  /**
   * Extract module name from file path or properties.
   */
  function extractModuleName(filePath: string, props: Record<string, unknown>): string {
    // Try to get from properties first
    const moduleName = getProp<string>(props, 'module_name')
    if (moduleName) {
      return moduleName
    }

    // Extract from file path
    if (filePath) {
      // Remove .py extension and convert path separators to dots
      const name = filePath
        .replace(/\.py$/, '')
        .replace(/\//g, '.')
        .replace(/\\/g, '.')

      // Get the last component or full path
      const parts = name.split('.')
      return parts[parts.length - 1] || name
    }

    return ''
  }

  /**
   * Clear the current inspector data.
   */
  function clearInspectorData(): void {
    inspectorDataRef.value = null
    error.value = null
    console.log('[useInspector] Cleared inspector data')
  }

  /**
   * Toggle a section's collapsed state.
   */
  function toggleSection(section: keyof InspectorSectionState): void {
    sectionState.value[section] = !sectionState.value[section]
  }

  /**
   * Format a decorator for display.
   */
  function formatDecorator(decorator: DecoratorInfo): string {
    if (decorator.source) {
      return decorator.source.startsWith('@') ? decorator.source : `@${decorator.source}`
    }

    let str = `@${decorator.name}`
    if (decorator.args && decorator.args.length > 0) {
      const argsStr = decorator.args
        .map((a) => (typeof a.key === 'string' ? `${a.key}=${a.value}` : a.value))
        .join(', ')
      str += `(${argsStr})`
    }

    return str
  }

  /**
   * Format a method signature for display.
   */
  function formatMethodSignature(method: MethodInfo): string {
    return method.signature
  }

  /**
   * Get the color for a decorator category.
   */
  function getDecoratorColor(category: DecoratorCategory): string {
    return DECORATOR_COLORS[category]
  }

  /**
   * Get the color for a Trinity type.
   */
  function getTrinityTypeColor(type: TrinityType): string {
    return TRINITY_TYPE_COLORS[type]
  }

  /**
   * Open source file in editor.
   * Uses the editor bridge for Tauri integration.
   */
  function handleOpenInEditor(source: SourceInfo): void {
    if (!source.file) {
      console.warn('[useInspector] No source file to open')
      return
    }

    // Use the editor bridge for Tauri integration
    openInEditor(source.file, source.line)
      .then((response) => {
        if (!response.success) {
          console.warn('[useInspector] Failed to open in editor:', response.message)
        }
      })
      .catch((err) => {
        console.error('[useInspector] Editor bridge error:', err)
      })

    // Also dispatch custom event for external editor integration
    const event = new CustomEvent('flowforge:open-in-editor', {
      detail: {
        file: source.file,
        line: source.line,
        column: source.column,
      },
      bubbles: true,
    })
    window.dispatchEvent(event)

    console.log('[useInspector] Open in editor:', source.file, source.line)
  }

  // ---------------------------------------------------------------------------
  // RETURN
  // ---------------------------------------------------------------------------

  return {
    inspectorData: readonly(inspectorData) as ComputedRef<InspectorData | null>,
    isLoading: readonly(isLoading) as Ref<boolean>,
    error: readonly(error) as Ref<string | null>,
    hasSelection,
    sectionState,
    fetchInspectorData,
    clearInspectorData,
    toggleSection,
    formatDecorator,
    formatMethodSignature,
    getDecoratorColor,
    getTrinityTypeColor,
    openInEditor: handleOpenInEditor,
  }
}

// Default export
export default useInspector
