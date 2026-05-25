/**
 * Link adapter - provides stubs for ComfyUI-specific link rendering
 * These can be implemented with FlowForge's own rendering system later
 */

import type { Point } from '../src/interfaces'

export interface LinkRenderContext {
  startPos: Point
  endPos: Point
  color: string
  thickness: number
}

/**
 * Litegraph link adapter - handles link rendering
 */
export class LitegraphLinkAdapter {
  static renderLink(
    ctx: CanvasRenderingContext2D,
    startPos: Point,
    endPos: Point,
    color: string = '#666',
    thickness: number = 2
  ): void {
    ctx.beginPath()
    ctx.strokeStyle = color
    ctx.lineWidth = thickness
    ctx.moveTo(startPos[0], startPos[1])

    // Simple bezier curve
    const midX = (startPos[0] + endPos[0]) / 2
    ctx.bezierCurveTo(
      midX, startPos[1],
      midX, endPos[1],
      endPos[0], endPos[1]
    )
    ctx.stroke()
  }
}
