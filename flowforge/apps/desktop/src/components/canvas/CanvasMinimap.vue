<template>
  <div class="minimap-container" v-show="visible">
    <div class="minimap-header">
      <span class="minimap-title">Minimap</span>
      <button class="minimap-close" title="Hide minimap" @click="$emit('toggle')">
        <i class="pi pi-times" />
      </button>
    </div>
    <canvas
      ref="minimapCanvas"
      class="minimap-canvas"
      :width="width"
      :height="height"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, type ShallowRef } from 'vue'
import type { LGraph } from '@/litegraph'
import type { LGraphCanvas } from '@/litegraph'

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<{
  graph: ShallowRef<LGraph | null>
  canvas: ShallowRef<LGraphCanvas | null>
  visible: boolean
}>()

defineEmits<{
  (e: 'toggle'): void
}>()

// =============================================================================
// CONSTANTS
// =============================================================================

const width = 200
const height = 150
const PADDING = 10

const NODE_COLORS: Record<string, string> = {
  component: '#4fc3f7',  // blue
  system: '#66bb6a',     // green
  resource: '#ab47bc',   // purple
  event: '#ffa726',      // orange
}
const DEFAULT_NODE_COLOR = '#888888'
const VIEWPORT_COLOR = 'rgba(255, 255, 255, 0.3)'
const VIEWPORT_BORDER = 'rgba(255, 255, 255, 0.7)'
const BG_COLOR = 'rgba(26, 26, 46, 0.85)'

// =============================================================================
// STATE
// =============================================================================

const minimapCanvas = ref<HTMLCanvasElement | null>(null)
let animFrameId: number | null = null
let isDragging = false

// =============================================================================
// DRAWING
// =============================================================================

function getNodeTrinityType(node: { type?: string; trinityType?: string; properties?: Record<string, unknown> }): string | null {
  if (node.trinityType) return node.trinityType as string
  if (node.properties?.trinityType) return node.properties.trinityType as string
  if (node.type) {
    const m = node.type.match(/^trinity\/(.+)$/)
    if (m) return m[1]
  }
  return null
}

function getNodeColor(node: { type?: string; trinityType?: string; properties?: Record<string, unknown> }): string {
  const tt = getNodeTrinityType(node)
  if (tt && NODE_COLORS[tt]) return NODE_COLORS[tt]
  return DEFAULT_NODE_COLOR
}

interface Bounds {
  minX: number; minY: number; maxX: number; maxY: number
  scaleX: number; scaleY: number; scale: number
  offsetX: number; offsetY: number
}

function computeBounds(): Bounds | null {
  const g = props.graph.value
  if (!g) return null
  const nodes = (g as unknown as { _nodes?: Array<{ pos: number[]; size: number[]; flags?: { hidden?: boolean } }> })._nodes
  if (!nodes || nodes.length === 0) return null

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const n of nodes) {
    if (n.flags?.hidden) continue
    minX = Math.min(minX, n.pos[0])
    minY = Math.min(minY, n.pos[1])
    maxX = Math.max(maxX, n.pos[0] + n.size[0])
    maxY = Math.max(maxY, n.pos[1] + n.size[1])
  }

  if (minX === Infinity) return null

  const graphW = maxX - minX || 1
  const graphH = maxY - minY || 1
  const drawW = width - PADDING * 2
  const drawH = height - PADDING * 2
  const scaleX = drawW / graphW
  const scaleY = drawH / graphH
  const scale = Math.min(scaleX, scaleY)
  const offsetX = PADDING + (drawW - graphW * scale) / 2
  const offsetY = PADDING + (drawH - graphH * scale) / 2

  return { minX, minY, maxX, maxY, scaleX, scaleY, scale, offsetX, offsetY }
}

function draw() {
  const el = minimapCanvas.value
  if (!el) return
  const ctx = el.getContext('2d')
  if (!ctx) return

  // Clear
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = BG_COLOR
  ctx.fillRect(0, 0, width, height)

  const g = props.graph.value
  const c = props.canvas.value
  if (!g || !c) return

  const bounds = computeBounds()
  if (!bounds) return

  const { minX, minY, scale, offsetX, offsetY } = bounds
  const nodes = (g as unknown as { _nodes?: Array<{ pos: number[]; size: number[]; type?: string; trinityType?: string; properties?: Record<string, unknown>; flags?: { hidden?: boolean } }> })._nodes
  if (!nodes) return

  // Draw nodes
  for (const n of nodes) {
    if (n.flags?.hidden) continue
    const x = offsetX + (n.pos[0] - minX) * scale
    const y = offsetY + (n.pos[1] - minY) * scale
    const w = Math.max(n.size[0] * scale, 3)
    const h = Math.max(n.size[1] * scale, 2)
    ctx.fillStyle = getNodeColor(n)
    ctx.fillRect(x, y, w, h)
  }

  // Draw viewport rect
  const ds = c.ds
  if (ds) {
    const canvasW = c.canvas?.width ?? 0
    const canvasH = c.canvas?.height ?? 0
    const canvasScale = ds.scale || 1
    const canvasOffset = ds.offset || [0, 0]

    // Visible area in graph coords
    const visMinX = -canvasOffset[0] / canvasScale
    const visMinY = -canvasOffset[1] / canvasScale
    const visMaxX = visMinX + canvasW / canvasScale
    const visMaxY = visMinY + canvasH / canvasScale

    // Convert to minimap coords
    const vx = offsetX + (visMinX - minX) * scale
    const vy = offsetY + (visMinY - minY) * scale
    const vw = (visMaxX - visMinX) * scale
    const vh = (visMaxY - visMinY) * scale

    ctx.fillStyle = VIEWPORT_COLOR
    ctx.fillRect(vx, vy, vw, vh)
    ctx.strokeStyle = VIEWPORT_BORDER
    ctx.lineWidth = 1.5
    ctx.strokeRect(vx, vy, vw, vh)
  }

  animFrameId = requestAnimationFrame(draw)
}

// =============================================================================
// NAVIGATION (click/drag to pan)
// =============================================================================

function navigateToMinimapPos(mx: number, my: number) {
  const c = props.canvas.value
  const bounds = computeBounds()
  if (!c || !bounds) return

  const { minX, minY, scale, offsetX, offsetY } = bounds
  const ds = c.ds
  if (!ds) return

  // Convert minimap coords to graph coords
  const graphX = minX + (mx - offsetX) / scale
  const graphY = minY + (my - offsetY) / scale

  const canvasW = c.canvas?.width ?? 0
  const canvasH = c.canvas?.height ?? 0
  const canvasScale = ds.scale || 1

  // Center viewport on clicked point
  ds.offset = [
    -(graphX * canvasScale) + canvasW / 2,
    -(graphY * canvasScale) + canvasH / 2,
  ]
  c.setDirty(true, true)
}

function getCanvasPos(e: PointerEvent): [number, number] {
  const el = minimapCanvas.value
  if (!el) return [0, 0]
  const rect = el.getBoundingClientRect()
  return [e.clientX - rect.left, e.clientY - rect.top]
}

function onPointerDown(e: PointerEvent) {
  isDragging = true
  const [mx, my] = getCanvasPos(e)
  navigateToMinimapPos(mx, my)
  ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
}

function onPointerMove(e: PointerEvent) {
  if (!isDragging) return
  const [mx, my] = getCanvasPos(e)
  navigateToMinimapPos(mx, my)
}

function onPointerUp(e: PointerEvent) {
  isDragging = false
  ;(e.target as HTMLElement).releasePointerCapture(e.pointerId)
}

// =============================================================================
// LIFECYCLE
// =============================================================================

onMounted(() => {
  animFrameId = requestAnimationFrame(draw)
})

onUnmounted(() => {
  if (animFrameId != null) {
    cancelAnimationFrame(animFrameId)
  }
})

watch(() => props.visible, (v) => {
  if (v && animFrameId == null) {
    animFrameId = requestAnimationFrame(draw)
  } else if (!v && animFrameId != null) {
    cancelAnimationFrame(animFrameId)
    animFrameId = null
  }
})
</script>

<style scoped>
.minimap-container {
  position: absolute;
  bottom: 16px;
  right: 16px;
  z-index: 100;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--border-color, #3d3d3d);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  pointer-events: auto;
}

.minimap-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 2px 8px;
  background-color: var(--panel-header-bg, #2d2d2d);
  border-bottom: 1px solid var(--border-color, #3d3d3d);
}

.minimap-title {
  font-size: 11px;
  color: var(--text-secondary, #cccccc);
  user-select: none;
}

.minimap-close {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--text-muted, #666);
  cursor: pointer;
  border-radius: 3px;
  transition: all 0.15s ease;
}

.minimap-close:hover {
  background-color: var(--hover-bg, #3a3a3a);
  color: var(--text-primary, #ffffff);
}

.minimap-close i {
  font-size: 10px;
}

.minimap-canvas {
  display: block;
  cursor: crosshair;
}
</style>
