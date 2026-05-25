# LiteGraph Minimap Implementation Guide for FlowForge
## 3.2.5 Research Phase Documentation

---

## Quick Reference

**Question: Does LiteGraph have a built-in minimap?**
**Answer: NO** - But the architecture is perfect for adding one.

---

## Key Findings

### What Exists
1. **Dual Canvas System**
   - Primary canvas (`this.canvas`): Main working area
   - Background canvas (`this.bgcanvas`): Separate render for groups and links
   - Both in `/src/litegraph/src/LGraphCanvas.ts`

2. **Rendering Hooks** (Best for minimap: `onDrawOverlay`)
   - `onDrawBackground(ctx, area)` - Behind all elements
   - `onDrawForeground(ctx, area)` - Above nodes/links
   - `onDrawOverlay(ctx)` - UI elements, NOT affected by zoom/pan
   - `onRender(canvas, ctx)` - Full render callback
   - `onRenderBackground(canvas, ctx)` - Background render alternative

3. **Viewport Tracking** (Perfect for minimap reference rect)
   - `canvas.ds.visible_area` - Rectangle of what's visible
   - `canvas.ds.offset` - Current pan [x, y]
   - `canvas.ds.scale` - Current zoom level
   - Located in `/src/litegraph/src/DragAndScale.ts`

4. **Graph Access**
   - `canvas.graph.nodes` - All nodes
   - `canvas.graph.links` - All connections
   - `canvas.visible_nodes` - Currently rendered nodes

### What's Missing
- ❌ No minimap
- ❌ No overview navigator
- ❌ No breadcrumb navigation
- ❌ No compass/orientation indicator

---

## Implementation Approach

### Recommended: Secondary Canvas Component

**File Structure:**
```
src/
├── components/
│   └── canvas/
│       ├── GraphCanvas.vue (EXISTING - add minimap reference)
│       └── GraphMinimap.vue (NEW - minimap component)
└── config/
    └── flowforge.config.ts (ADD minimap config)
```

### Key API Access Points

```typescript
// From GraphCanvas.vue, pass to Minimap component:
const canvas: LGraphCanvas = canvasRef.value

// Access needed by minimap:
canvas.graph.nodes              // All nodes
canvas.graph.links              // All connections
canvas.ds.visible_area          // Current viewport [x, y, w, h]
canvas.ds.scale                 // Current zoom level
canvas.ds.offset               // Current pan offset [x, y]
canvas.graph.getBoundingBox()  // Full graph bounds

// Available from DragAndScale for animations:
canvas.ds.scale                 // Get/set zoom
canvas.ds.offset               // Get/set pan position
```

### Coordinate Transformation

```typescript
// Graph to Canvas (what user sees)
function graphToCanvas([gx, gy], scale, offset) {
  const cx = gx * scale - offset[0] * scale
  const cy = gy * scale - offset[1] * scale
  return [cx, cy]
}

// Canvas to Graph (world coordinates)
function canvasToGraph([cx, cy], scale, offset) {
  const gx = (cx + offset[0] * scale) / scale
  const gy = (cy + offset[1] * scale) / scale
  return [gx, gy]
}

// Minimap to Canvas (from minimap click to pan location)
function minimapToGraph(minimapX, minimapY, graphBounds, minimapSize) {
  // minimapX/Y are pixel coords in minimap canvas
  // Convert to graph space by scaling back up
  const scale = graphBounds.width / minimapSize.width
  const gx = graphBounds[0] + minimapX * scale
  const gy = graphBounds[1] + minimapY * scale
  return [gx, gy]
}
```

---

## Code Snippets for Implementation

### 1. Add to GraphCanvas.vue

```vue
<script setup>
// After canvas.value initialization:
const minimapRef = ref<InstanceType<typeof GraphMinimap> | null>(null)

onMounted(async () => {
  await initializeCanvas()
  // After canvas ready, set up minimap reference
  if (minimapRef.value && canvas.value) {
    minimapRef.value.setCanvasInstance(canvas.value)
  }
})

// Listen for viewport changes
function setupEventListeners() {
  // ... existing listeners ...

  // Update minimap when graph changes
  graph.value?.addEventListener('change', () => {
    minimapRef.value?.update()
  })
}
</script>

<template>
  <div ref="containerEl" class="graph-canvas-container">
    <canvas ref="canvasEl" class="litegraph-canvas"></canvas>
    <GraphMinimap
      ref="minimapRef"
      class="graph-minimap"
    />
    <div v-if="!isReady" class="canvas-loading">
      <span>Initializing canvas...</span>
    </div>
  </div>
</template>

<style scoped>
.graph-canvas-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.graph-minimap {
  position: absolute;
  bottom: 20px;
  right: 20px;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}
</style>
```

### 2. New Component: GraphMinimap.vue Structure

```vue
<template>
  <div class="minimap-container" :class="{ 'minimap-collapsed': isCollapsed }">
    <div class="minimap-header">
      <span class="minimap-title">Overview</span>
      <button
        class="minimap-toggle"
        @click="toggleCollapse"
        :title="isCollapsed ? 'Expand' : 'Collapse'"
      >
        {{ isCollapsed ? '↗' : '↙' }}
      </button>
    </div>
    <canvas
      ref="minimapCanvas"
      class="minimap-canvas"
      :width="MIN_MAP_WIDTH"
      :height="MIN_MAP_HEIGHT"
      @click="handleMinimapClick"
      @mousemove="handleMinimapHover"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import type { LGraphCanvas } from '@/litegraph'

const MIN_MAP_WIDTH = 200
const MIN_MAP_HEIGHT = 150
const PADDING = 5

const minimapCanvas = ref<HTMLCanvasElement | null>(null)
const isCollapsed = ref(false)
let canvas: LGraphCanvas | null = null
let animationId: number | null = null

// Exposed methods
function setCanvasInstance(lgraphCanvas: LGraphCanvas) {
  canvas = lgraphCanvas
  startRendering()
}

function update() {
  if (minimapCanvas.value && canvas) {
    renderMinimap()
  }
}

function toggleCollapse() {
  isCollapsed.value = !isCollapsed.value
}

// Rendering
function renderMinimap() {
  if (!minimapCanvas.value || !canvas || !canvas.graph) return

  const ctx = minimapCanvas.value.getContext('2d')
  if (!ctx) return

  const graph = canvas.graph
  const nodes = graph.nodes

  // Get graph bounds
  const bounds = graph.getBoundingBox()
  const graphW = bounds[2] - bounds[0] || 100
  const graphH = bounds[3] - bounds[1] || 100

  // Calculate minimap scale
  const mapW = MIN_MAP_WIDTH - PADDING * 2
  const mapH = MIN_MAP_HEIGHT - PADDING * 2
  const scale = Math.min(mapW / graphW, mapH / graphH)

  // Clear canvas
  ctx.fillStyle = '#222'
  ctx.fillRect(0, 0, MIN_MAP_WIDTH, MIN_MAP_HEIGHT)

  // Draw border
  ctx.strokeStyle = '#444'
  ctx.lineWidth = 1
  ctx.strokeRect(0, 0, MIN_MAP_WIDTH, MIN_MAP_HEIGHT)

  // Draw nodes
  ctx.fillStyle = '#666'
  for (const node of nodes) {
    const x = (node.pos[0] - bounds[0]) * scale + PADDING
    const y = (node.pos[1] - bounds[1]) * scale + PADDING
    const w = Math.max(node.size[0] * scale, 4)
    const h = Math.max(node.size[1] * scale, 4)

    ctx.fillRect(x, y, w, h)
  }

  // Draw current viewport rectangle
  const visArea = canvas.ds.visible_area
  const vpX = (visArea[0] - bounds[0]) * scale + PADDING
  const vpY = (visArea[1] - bounds[1]) * scale + PADDING
  const vpW = visArea[2] * scale
  const vpH = visArea[3] * scale

  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 2
  ctx.strokeRect(vpX, vpY, vpW, vpH)

  ctx.fillStyle = 'rgba(255, 255, 255, 0.1)'
  ctx.fillRect(vpX, vpY, vpW, vpH)
}

function startRendering() {
  const render = () => {
    renderMinimap()
    animationId = requestAnimationFrame(render)
  }
  animationId = requestAnimationFrame(render)
}

function handleMinimapClick(e: MouseEvent) {
  if (!minimapCanvas.value || !canvas || !canvas.graph) return

  const rect = minimapCanvas.value.getBoundingClientRect()
  const minimapX = e.clientX - rect.left - PADDING
  const minimapY = e.clientY - rect.top - PADDING

  // Calculate graph bounds
  const graph = canvas.graph
  const bounds = graph.getBoundingBox()
  const graphW = bounds[2] - bounds[0] || 100
  const graphH = bounds[3] - bounds[1] || 100

  // Calculate scale
  const mapW = MIN_MAP_WIDTH - PADDING * 2
  const mapH = MIN_MAP_HEIGHT - PADDING * 2
  const scale = Math.min(mapW / graphW, mapH / graphH)

  // Convert minimap coords to graph coords
  const graphX = bounds[0] + minimapX / scale
  const graphY = bounds[1] + minimapY / scale

  // Pan to that location
  const visArea = canvas.ds.visible_area
  canvas.ds.offset[0] = graphX - visArea[2] / 2
  canvas.ds.offset[1] = graphY - visArea[3] / 2
  canvas.setDirty(true, true)
}

function handleMinimapHover(e: MouseEvent) {
  if (!minimapCanvas.value) return
  minimapCanvas.value.style.cursor = 'pointer'
}

onMounted(() => {
  // Component ready
})

defineExpose({
  setCanvasInstance,
  update,
  toggleCollapse
})
</script>

<style scoped>
.minimap-container {
  display: flex;
  flex-direction: column;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 4px;
  overflow: hidden;
  font-size: 12px;
  user-select: none;
}

.minimap-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 8px;
  background: #222;
  border-bottom: 1px solid #333;
}

.minimap-title {
  color: #aaa;
  font-weight: 500;
  font-size: 11px;
}

.minimap-toggle {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  padding: 0;
  font-size: 12px;
}

.minimap-toggle:hover {
  color: #aaa;
}

.minimap-canvas {
  display: block;
  background: #1a1a1a;
  cursor: grab;
}

.minimap-canvas:active {
  cursor: grabbing;
}

.minimap-collapsed .minimap-canvas {
  display: none;
}
</style>
```

### 3. Configuration Addition

```typescript
// In flowforge.config.ts
export const MINIMAP_CONFIG = {
  enabled: true,
  position: 'bottom-right' as const,
  width: 200,
  height: 150,
  padding: 5,
  colors: {
    background: '#1a1a1a',
    border: '#333',
    node: '#666',
    viewport: '#fff',
    viewportAlpha: 0.1
  },
  showHeader: true,
  collapsible: true,
  defaultCollapsed: false
}
```

---

## Integration Steps

1. **Create Minimap Component**
   - Add `/src/components/canvas/GraphMinimap.vue`
   - Implement basic rendering (nodes as rectangles)

2. **Connect to GraphCanvas.vue**
   - Import minimap component
   - Add minimap ref to template
   - Pass canvas instance on mount
   - Listen for graph changes

3. **Add Styling**
   - Match dark theme
   - Position in bottom-right corner
   - Add shadow/border for visibility

4. **Implement Interactivity**
   - Click-to-pan
   - Hover effects
   - Collapse/expand button

5. **Configuration**
   - Add to flowforge.config
   - Make position/size configurable
   - Add color customization

6. **Polish** (Optional Phase 2)
   - Drag-to-pan from minimap
   - Node color by type
   - Zoom level indicator
   - Performance optimizations

---

## Testing Checklist

- [ ] Minimap renders with graph visible
- [ ] Viewport rectangle updates with pan
- [ ] Viewport rectangle updates with zoom
- [ ] Click in minimap pans to that location
- [ ] Collapse/expand button works
- [ ] Performance acceptable with large graphs (100+ nodes)
- [ ] Works with Trinity ECS node colors
- [ ] Mobile responsive (if needed)
- [ ] Dark theme integration

---

## Performance Notes

- Render at ~30fps (not needed every frame)
- Only redraw on graph change (not continuous)
- Use separate render cycle from main canvas
- Keep node rendering simple (rectangles only)
- Consider debouncing pan animations

---

## File References

**Existing Files to Reference:**
- `/src/litegraph/src/LGraphCanvas.ts` - Canvas hooks (line 611-762)
- `/src/litegraph/src/DragAndScale.ts` - Viewport/scale management
- `/src/components/canvas/GraphCanvas.vue` - Integration point
- `/src/config/flowforge.config.ts` - Configuration

**Files to Create:**
- `/src/components/canvas/GraphMinimap.vue` - NEW component
- `/src/RESEARCH_MINIMAP_3.2.5.md` - Research documentation (created)
- `/src/MINIMAP_IMPLEMENTATION_GUIDE.md` - This file

---

## Related Issues & Features

This minimap could enhance:
- Large graph navigation
- Node discovery
- Layout understanding
- Visual hierarchy awareness

Could be combined with:
- Breadcrumb navigation (path to current subgraph)
- Node search/focus
- Execution timeline visualization
- Group/frame visualization

---

## Conclusion

**Minimap is definitely feasible** for FlowForge using the existing LiteGraph architecture. No modifications to core LiteGraph needed - purely additive via Vue component + canvas hooks.

Implementation effort: ~2-4 hours for basic version, ~6-8 hours for fully polished version.
