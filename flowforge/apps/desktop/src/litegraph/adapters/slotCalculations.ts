/**
 * Slot calculation adapter - provides stubs for ComfyUI-specific slot positioning
 * These can be implemented with FlowForge's own rendering system later
 */

import type { Point } from '../src/interfaces'

export interface SlotPositionContext {
  nodeId: number
  slotIndex: number
  isInput: boolean
}

/**
 * Calculate the position of a slot
 */
export function getSlotPosition(
  _context: SlotPositionContext,
  nodePos: Point,
  nodeSize: [number, number],
  slotIndex: number,
  isInput: boolean,
  _titleHeight: number = 30
): Point {
  const slotSpacing = 20
  const slotStartY = 30 // Below title
  const y = nodePos[1] + slotStartY + slotIndex * slotSpacing
  const x = isInput ? nodePos[0] : nodePos[0] + nodeSize[0]
  return [x, y]
}

/**
 * Calculate input slot position from slot data
 */
export function calculateInputSlotPosFromSlot(
  nodePos: Point,
  nodeSize: [number, number],
  slotIndex: number,
  _titleHeight: number = 30
): Point {
  return getSlotPosition({} as SlotPositionContext, nodePos, nodeSize, slotIndex, true)
}
