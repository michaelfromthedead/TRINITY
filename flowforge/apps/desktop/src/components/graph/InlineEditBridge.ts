/**
 * Inline Edit Bridge
 *
 * Provides event-based communication between LiteGraph nodes and Vue components
 * for inline editing of node names, field types, and default values.
 *
 * LiteGraph nodes dispatch custom events when editing should start,
 * and Vue components listen for these events to show inline editors.
 */

// =============================================================================
// Event Names
// =============================================================================

/** Event dispatched when inline editing should start */
export const INLINE_EDIT_START_EVENT = 'flowforge:inline-edit-start'

/** Event dispatched when inline editing is committed */
export const INLINE_EDIT_COMMIT_EVENT = 'flowforge:inline-edit-commit'

/** Event dispatched when inline editing is cancelled */
export const INLINE_EDIT_CANCEL_EVENT = 'flowforge:inline-edit-cancel'

/** Event dispatched when a field type selection should start */
export const TYPE_SELECT_START_EVENT = 'flowforge:type-select-start'

// =============================================================================
// Event Detail Types
// =============================================================================

/**
 * Detail for inline edit start event.
 */
export interface InlineEditStartDetail {
  /** ID of the node being edited */
  nodeId: string
  /** Type of edit: name (class name), type (field type), or default (default value) */
  editType: 'name' | 'type' | 'default'
  /** Name of the field (for type and default edits) */
  fieldName?: string
  /** Current value before editing */
  currentValue: string
  /** Position for the editor overlay (canvas coordinates) */
  position: {
    x: number
    y: number
    width: number
    height: number
  }
  /** Font size for the editor */
  fontSize?: number
}

/**
 * Detail for inline edit commit event.
 */
export interface InlineEditCommitDetail {
  /** ID of the node that was edited */
  nodeId: string
  /** Type of edit that was committed */
  editType: 'name' | 'type' | 'default'
  /** Name of the field (for type and default edits) */
  fieldName?: string
  /** Old value before the edit */
  oldValue: string
  /** New value after the edit */
  newValue: string
}

/**
 * Detail for inline edit cancel event.
 */
export interface InlineEditCancelDetail {
  /** ID of the node that was being edited */
  nodeId: string
  /** Type of edit that was cancelled */
  editType: 'name' | 'type' | 'default'
  /** Name of the field (for type and default edits) */
  fieldName?: string
}

/**
 * Detail for type selection start event.
 */
export interface TypeSelectStartDetail {
  /** ID of the node being edited */
  nodeId: string
  /** Name of the field being edited */
  fieldName: string
  /** Current type value */
  currentType: string
  /** Position for the type selector dropdown */
  position: {
    x: number
    y: number
    width: number
  }
}

// =============================================================================
// Event Dispatch Functions
// =============================================================================

/**
 * Dispatch an event to start inline editing.
 *
 * Call this from LiteGraph node event handlers (onDblClick, onMouseDown, etc.)
 * when the user should be able to edit a value inline.
 *
 * @param detail - The edit start details
 */
export function dispatchInlineEditStart(detail: InlineEditStartDetail): void {
  window.dispatchEvent(
    new CustomEvent(INLINE_EDIT_START_EVENT, { detail })
  )
}

/**
 * Dispatch an event when inline editing is committed.
 *
 * @param detail - The edit commit details
 */
export function dispatchInlineEditCommit(detail: InlineEditCommitDetail): void {
  window.dispatchEvent(
    new CustomEvent(INLINE_EDIT_COMMIT_EVENT, { detail })
  )
}

/**
 * Dispatch an event when inline editing is cancelled.
 *
 * @param detail - The edit cancel details
 */
export function dispatchInlineEditCancel(detail: InlineEditCancelDetail): void {
  window.dispatchEvent(
    new CustomEvent(INLINE_EDIT_CANCEL_EVENT, { detail })
  )
}

/**
 * Dispatch an event to start type selection.
 *
 * @param detail - The type selection details
 */
export function dispatchTypeSelectStart(detail: TypeSelectStartDetail): void {
  window.dispatchEvent(
    new CustomEvent(TYPE_SELECT_START_EVENT, { detail })
  )
}

// =============================================================================
// Hit Testing Utilities
// =============================================================================

/**
 * Calculate the bounding box of a node's title/class name area.
 *
 * @param nodePos - Node position [x, y]
 * @param nodeSize - Node size [width, height]
 * @param headerHeight - Height of the header area
 * @returns Bounding box for the title area
 */
export function getTitleBounds(
  nodePos: [number, number],
  nodeSize: [number, number],
  headerHeight: number
): { x: number; y: number; width: number; height: number } {
  return {
    x: nodePos[0],
    y: nodePos[1],
    width: nodeSize[0],
    height: headerHeight
  }
}

/**
 * Calculate the bounding box of a field's type area.
 *
 * @param nodePos - Node position [x, y]
 * @param nodeSize - Node size [width, height]
 * @param fieldIndex - Index of the field
 * @param fieldY - Y offset of the field within the node
 * @param fieldHeight - Height of a field row
 * @param typeX - X offset where type text starts
 * @param typeWidth - Width of the type text
 * @returns Bounding box for the type area
 */
export function getFieldTypeBounds(
  nodePos: [number, number],
  fieldY: number,
  fieldHeight: number,
  typeX: number,
  typeWidth: number
): { x: number; y: number; width: number; height: number } {
  return {
    x: nodePos[0] + typeX,
    y: nodePos[1] + fieldY,
    width: typeWidth,
    height: fieldHeight
  }
}

/**
 * Calculate the bounding box of a field's default value area.
 *
 * @param nodePos - Node position [x, y]
 * @param fieldY - Y offset of the field within the node
 * @param fieldHeight - Height of a field row
 * @param defaultX - X offset where default value text starts
 * @param defaultWidth - Width of the default value text
 * @returns Bounding box for the default value area
 */
export function getFieldDefaultBounds(
  nodePos: [number, number],
  fieldY: number,
  fieldHeight: number,
  defaultX: number,
  defaultWidth: number
): { x: number; y: number; width: number; height: number } {
  return {
    x: nodePos[0] + defaultX,
    y: nodePos[1] + fieldY,
    width: defaultWidth,
    height: fieldHeight
  }
}

/**
 * Convert canvas coordinates to screen coordinates.
 *
 * @param canvasX - X coordinate in canvas space
 * @param canvasY - Y coordinate in canvas space
 * @param canvasOffset - Canvas offset [offsetX, offsetY]
 * @param canvasScale - Canvas scale factor
 * @param canvasRect - Bounding rect of the canvas element
 * @returns Screen coordinates { x, y }
 */
export function canvasToScreen(
  canvasX: number,
  canvasY: number,
  canvasOffset: [number, number],
  canvasScale: number,
  canvasRect: DOMRect
): { x: number; y: number } {
  return {
    x: canvasRect.left + (canvasX + canvasOffset[0]) * canvasScale,
    y: canvasRect.top + (canvasY + canvasOffset[1]) * canvasScale
  }
}

/**
 * Convert a bounding box from canvas coordinates to screen coordinates.
 *
 * @param bounds - Bounding box in canvas coordinates
 * @param canvasOffset - Canvas offset [offsetX, offsetY]
 * @param canvasScale - Canvas scale factor
 * @param canvasRect - Bounding rect of the canvas element
 * @returns Bounding box in screen coordinates
 */
export function boundsToScreen(
  bounds: { x: number; y: number; width: number; height: number },
  canvasOffset: [number, number],
  canvasScale: number,
  canvasRect: DOMRect
): { x: number; y: number; width: number; height: number } {
  const topLeft = canvasToScreen(bounds.x, bounds.y, canvasOffset, canvasScale, canvasRect)
  return {
    x: topLeft.x,
    y: topLeft.y,
    width: bounds.width * canvasScale,
    height: bounds.height * canvasScale
  }
}
