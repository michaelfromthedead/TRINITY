# LiteGraph Minimap Feature Research Report
## Research Date: 2026-01-27
## Focus: FlowForge Desktop Application

---

## EXECUTIVE SUMMARY

**Minimap Status: NOT BUILT-IN**

LiteGraph does NOT have a native minimap/overview feature. However, the architecture provides excellent hooks and APIs to implement one.

---

## 1. LITEGRAPH ARCHITECTURE ANALYSIS

### Canvas Rendering System
Located: `/src/litegraph/src/LGraphCanvas.ts` (8,639 lines)

**Key Components:**
- **Primary Canvas** (`this.canvas`): Main interactive canvas where users work
- **Background Canvas** (`this.bgcanvas`): Separate canvas for rendering graph structure (groups, connections)
- **Drawing Architecture**: 
  - `draw()` method (line 4670): Main rendering orchestrator
  - `drawFrontCanvas()` (line 4716): Renders nodes and interactions
  - `drawBackCanvas()` (line 5090): Renders static elements

### Viewport Management
Located: `/src/litegraph/src/DragAndScale.ts`

**Key Features:**
- `DragAndScaleState`: Manages offset and scale
  - `offset`: Canvas-to-graph translation
  - `scale`: Current zoom level
- `computeVisibleArea()` (line 95): Calculates visible portion of graph
- `visible_area`: Rectangle object showing what's currently viewable
- Full animation support with easing functions

---

## 2. EXTENSION/HOOK SYSTEM

LiteGraph provides multiple hooks for custom rendering without modifying core:

### Canvas-Level Hooks (LGraphCanvas)
```typescript
onDrawBackground?: (ctx: CanvasRenderingContext2D, area: Rect) => void
onDrawForeground?: (ctx: CanvasRenderingContext2D, area: Rect) => void
onDrawOverlay?: (ctx: CanvasRenderingContext2D) => void
onRender?: (canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) => void
onRenderBackground?: (canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) => boolean
onDrawLinkTooltip?: (ctx: CanvasRenderingContext2D, link: LLink | null, canvas: LGraphCanvas) => boolean
```

**Best Hook for Minimap: `onDrawOverlay`**
- Called after all graph elements rendered
- Context is NOT affected by current transform/zoom
- Perfect for UI elements like minimap

### Node-Level Hooks
- `node.onDrawBackground()`: Render behind node
- `node.onDrawForeground()`: Render on top of node
- `node.onDrawCollapsed()`: Custom collapsed state rendering

---

## 3. WHAT EXISTS IN FLOWFORGE

### Current Implementation
File: `/src/components/canvas/GraphCanvas.vue`

**Canvas Configuration:**
```typescript
canvas.value.background_color = CANVAS_CONFIG.backgroundColor
canvas.value.render_canvas_border = false
canvas.value.render_shadows = true
canvas.value.render_curved_connections = true
canvas.value.render_connection_arrows = 'middle_right'
canvas.value.allow_searchbox = true
canvas.value.allow_dragnodes = true
```

**Available Properties (No Minimap):**
- Node rendering and interaction
- Connection rendering
- Background canvas for static elements
- Full zoom and pan capabilities
- Viewport tracking via DragAndScale

---

## 4. KEY DATA FOR MINIMAP IMPLEMENTATION

### Access Points Available
```typescript
// From LGraphCanvas instance:
canvas.visible_nodes       // Array of currently visible nodes
canvas.ds.offset          // Current pan offset [x, y]
canvas.ds.scale           // Current zoom level
canvas.ds.visible_area    // Rectangle of visible area
canvas.graph              // The LGraph instance with all nodes/links
canvas.canvas             // The main DOM canvas element
canvas.bgcanvas           // Background canvas (separate render)

// From LGraph instance:
graph.nodes               // All nodes in graph
graph.links               // All connections
graph._groups             // Node groups
graph.canvas              // Computed bounds
```

### Critical Methods
```typescript
// Get all node bounds
graph.getVisibleNodes()       // Only currently visible ones
graph.nodes                   // All nodes in graph

// Transform between coordinate systems
canvas.convertEventToCanvasOffset(event)
canvas.convertCanvasToGraphCoordinates([x, y])
canvas.convertGraphToCanvasCoordinates([x, y])
```

---

## 5. MINIMAP IMPLEMENTATION APPROACHES

### Approach 1: Secondary Small Canvas (Recommended)
**Pros:**
- Clean separation of concerns
- Easy to position (corner of screen)
- Can be toggled on/off
- Minimal performance impact

**Implementation Points:**
- Create small canvas element (e.g., 200x150px)
- Use `onDrawOverlay` hook to render minimap UI
- Scale down entire graph (divide by 4-10x)
- Draw current viewport rectangle as overlay
- Handle click-to-pan functionality

**Code Location:** Would go in `GraphCanvas.vue` or new `Minimap.vue` component

### Approach 2: Overlay on Main Canvas
**Pros:**
- Single canvas, no extra DOM elements
- Part of main render loop

**Cons:**
- More complex coordinate management
- Must account for all transforms

**Code Location:** Would hook into `onDrawOverlay` in LGraphCanvas

### Approach 3: Inset in Corner (ComfyUI Style)
**Pros:**
- Always visible
- Doesn't take up extra space
- Professional appearance

**Implementation:**
- Use `onDrawOverlay` for corner positioning
- Render small scaled version of graph
- Add viewport indicator rectangle
- Make interactive for navigation

---

## 6. TECHNICAL REQUIREMENTS

### Rendering
- Access to graph structure via `canvas.graph`
- Access to visible area via `canvas.ds.visible_area`
- Access to zoom/pan state via `canvas.ds.offset` and `canvas.ds.scale`

### Interaction
- Mouse event handling for minimap clicks
- Coordinate conversion between minimap and canvas
- Pan animation to clicked location (via `canvas.ds`)

### Performance
- Scale calculation to fit entire graph in minimap
- Efficient rendering of node rectangles (not full rendering)
- Update only when graph changes (use dirty flags)

---

## 7. LITEGRAPH NATIVE FEATURES (Existing)

The following features ARE built-in:
- ✅ Zoom in/out (mouse wheel, Ctrl+Shift+Drag)
- ✅ Pan (middle mouse drag)
- ✅ Fit to view (animate to bounding box)
- ✅ Viewport awareness (knows what's visible)
- ✅ Scale/offset tracking
- ✅ Node visibility culling (only renders visible nodes)
- ✅ Grid snapping
- ✅ Node groups/frames
- ✅ Execution order visualization

The following do NOT exist:
- ❌ Minimap/Overview panel
- ❌ Timeline/History view
- ❌ Breadcrumb navigation
- ❌ In-canvas compass/orientation indicator

---

## 8. RECOMMENDATIONS

### For FlowForge Implementation

**Phase 1: Basic Minimap**
1. Create `/src/components/canvas/GraphMinimap.vue`
2. Add small canvas element (200x150px) to bottom-right
3. Implement rendering via:
   - Get graph bounds from `canvas.graph`
   - Calculate scale to fit in minimap
   - Use `onDrawOverlay` hook OR separate component render loop
   - Draw simplified node rectangles
   - Draw viewport indicator (semi-transparent rectangle)

**Phase 2: Interactive**
1. Add click handler to minimap
2. Convert minimap coordinates to graph coordinates
3. Animate pan to that location using `canvas.ds`

**Phase 3: Polish**
1. Toggle visibility button
2. Drag to pan from minimap
3. Right-click options
4. Zoom indicator overlay
5. Color coding by node type (Trinity nodes)

### Integration Points
- File: `GraphCanvas.vue` - Add minimap component and hooks
- File: `Minimap.vue` (new) - Minimap component
- Config: `flowforge.config` - Minimap position, size, colors

### Performance Considerations
- Use `requestAnimationFrame` for minimap updates
- Only redraw when graph changes
- Keep minimap rendering lightweight (no node details)
- Use separate offscreen canvas if needed

---

## 9. DEPENDENCIES & IMPORTS

Required from litegraph:
```typescript
import { LGraph, LGraphCanvas } from '@/litegraph'
import type { LGraphNode } from '@/litegraph'
```

Access patterns:
```typescript
// From GraphCanvas.vue reference to canvas instance
const canvas: LGraphCanvas = canvasRef.value
const graph: LGraph = canvas.graph

// Properties needed
const nodes = graph.nodes
const visible = canvas.ds.visible_area
const scale = canvas.ds.scale
const offset = canvas.ds.offset
```

---

## 10. COMPARISON WITH COMFYUI

ComfyUI (the origin of this LiteGraph fork) does NOT have a built-in minimap either. Any minimap in ComfyUI instances would be:
- Custom extensions
- Third-party plugins
- Not part of core LiteGraph

This FlowForge minimap would be a **competitive advantage**.

---

## CONCLUSION

**LiteGraph Status:** No native minimap feature exists.

**Feasibility:** Highly feasible - the architecture supports it perfectly.

**Effort:** Medium complexity
- UI: Low (new Vue component)
- Logic: Medium (coordinate transformations)
- Integration: Easy (uses onDrawOverlay hook)

**Recommended Path:** Approach #1 (Secondary Canvas in Corner)
- Clean implementation
- Easy to maintain
- Professional appearance
- Standard in node-graph editors

**Next Steps:**
1. Implement basic minimap with viewport indicator
2. Add click-to-pan interaction
3. Add drag-to-pan from minimap
4. Style to match Trinity ECS theme
5. Add configuration options

