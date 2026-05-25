/**
 * Layout adapter - provides stubs for ComfyUI-specific layout functionality
 * These can be implemented with FlowForge's own layout system later
 */

export enum LayoutSource {
  Node = 'node',
  Canvas = 'canvas',
  User = 'user',
  System = 'system'
}

export interface Bounds {
  left: number
  top: number
  right: number
  bottom: number
}

/**
 * Stub for layout mutations - returns no-op functions
 * Replace with FlowForge's layout system implementation
 */
export function useLayoutMutations() {
  return {
    setNodePosition: (_nodeId: number, _x: number, _y: number, _source?: LayoutSource) => {},
    setNodeSize: (_nodeId: number, _width: number, _height: number, _source?: LayoutSource) => {},
    setGroupPosition: (_groupId: number, _x: number, _y: number, _source?: LayoutSource) => {},
    setGroupSize: (_groupId: number, _width: number, _height: number, _source?: LayoutSource) => {},
    setReroutePosition: (_rerouteId: number, _x: number, _y: number, _source?: LayoutSource) => {},
    beginBatch: () => {},
    endBatch: () => {}
  }
}

/**
 * Stub for layout store
 */
export const layoutStore = {
  getNodeBounds: (_nodeId: number): Bounds | null => null,
  getGroupBounds: (_groupId: number): Bounds | null => null,
  getRerouteBounds: (_rerouteId: number): Bounds | null => null
}

/**
 * Stub for removing node title height from size calculations
 */
export function removeNodeTitleHeight(height: number, _titleHeight?: number): number {
  return height
}
