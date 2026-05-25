# Minimap Feature - Quick Reference Card

**Status:** NOT BUILT-IN (but highly feasible to implement)

---

## One-Line Summary

LiteGraph has NO minimap, but provides perfect hooks and data access for a Vue component-based implementation with ~6-8 hours effort.

---

## Where to Find Information

| Need | Document | Time |
|------|----------|------|
| Executive overview | `RESEARCH_SUMMARY_3.2.5.txt` | 10 min |
| Technical details | `RESEARCH_MINIMAP_3.2.5.md` | 30 min |
| **Start coding** | **`MINIMAP_IMPLEMENTATION_GUIDE.md`** | 45 min |
| This reference | This file | 2 min |

---

## Critical Data Access Points

```typescript
// From LGraphCanvas instance
canvas.graph.nodes           // All nodes in graph
canvas.graph.getBoundingBox() // Full extent [x1, y1, x2, y2]
canvas.ds.visible_area       // Current viewport [x, y, w, h]
canvas.ds.offset             // Pan position [x, y]
canvas.ds.scale              // Zoom level (1.0 = 100%)
```

---

## Key Code Patterns

### Render Minimap (Pseudocode)
```typescript
const graph = canvas.graph
const bounds = graph.getBoundingBox()
const scale = minimapSize / max(bounds.width, bounds.height)

// Draw nodes
for (const node of graph.nodes) {
  ctx.fillRect(
    (node.pos[0] - bounds[0]) * scale + padding,
    (node.pos[1] - bounds[1]) * scale + padding,
    node.size[0] * scale,
    node.size[1] * scale
  )
}

// Draw viewport indicator
const viewport = canvas.ds.visible_area
ctx.strokeRect(
  (viewport[0] - bounds[0]) * scale + padding,
  (viewport[1] - bounds[1]) * scale + padding,
  viewport[2] * scale,
  viewport[3] * scale
)
```

### Pan to Location
```typescript
// User clicked minimap at [mapX, mapY]
const graphX = bounds[0] + (mapX - padding) * (bounds.width / minimapWidth)
const graphY = bounds[1] + (mapY - padding) * (bounds.height / minimapHeight)

canvas.ds.offset[0] = graphX - visibleArea[2] / 2
canvas.ds.offset[1] = graphY - visibleArea[3] / 2
canvas.setDirty(true, true)
```

---

## Implementation Checklist

### Phase 1 - Basic (2-3 hours)
- [ ] Create `src/components/canvas/GraphMinimap.vue`
- [ ] Implement canvas rendering
- [ ] Draw nodes as rectangles
- [ ] Draw viewport indicator
- [ ] Integrate into `GraphCanvas.vue`
- [ ] Test basic functionality

### Phase 2 - Interactive (1-2 hours)
- [ ] Implement click-to-pan
- [ ] Add drag-to-pan
- [ ] Smooth animation
- [ ] Collapse/expand toggle

### Phase 3 - Polish (2-3 hours)
- [ ] Node type coloring
- [ ] Configuration options
- [ ] Performance optimization
- [ ] Documentation

---

## Files to Create

```
src/components/canvas/
└── GraphMinimap.vue          NEW - Minimap component

src/config/
└── minimap.config.ts         NEW - Configuration
```

## Files to Modify

```
src/components/canvas/
└── GraphCanvas.vue           - Add minimap import & integration
```

## Reference Files (Don't Modify)

```
src/litegraph/src/
├── LGraphCanvas.ts           - Canvas implementation
└── DragAndScale.ts           - Viewport management
```

---

## Rendering Hooks Available

```typescript
// In LGraphCanvas - pick ONE for minimap overlay
canvas.onDrawBackground()     // Behind everything
canvas.onDrawForeground()     // Above nodes
canvas.onDrawOverlay()        // UI elements (BEST CHOICE)
canvas.onRender()             // Full render
```

**Recommendation:** Use separate Vue component with its own canvas instead of hooks (cleaner code).

---

## Performance Notes

- Separate render pass: ~30fps (sufficient)
- Update only on graph change
- Keep node rendering simple (rectangles only)
- Minimal DOM elements
- Expected <5ms per frame

---

## Testing Quick Checklist

```
Functional:
- [ ] Minimap renders all nodes
- [ ] Viewport box updates on pan
- [ ] Viewport box updates on zoom
- [ ] Click in minimap pans to location
- [ ] No visible lag or stutter

Performance:
- [ ] Smooth 30fps rendering
- [ ] No frame drops on zoom
- [ ] Large graphs (100+ nodes) OK
- [ ] No memory leak on repeated pan/zoom

Integration:
- [ ] Works in GraphCanvas component
- [ ] Respects dark theme colors
- [ ] Collapse/expand works
- [ ] No console errors
```

---

## Common Gotchas

1. **Coordinate System:** Graph ≠ Canvas pixels
   - Convert using scale and offset
   - See IMPLEMENTATION_GUIDE for formulas

2. **Viewport Rectangle:** Use `visible_area`, not screen bounds
   - `visible_area[2]` and `[3]` are width/height
   - Not absolute coordinates

3. **Animation:** Use `canvas.ds` directly, then `setDirty()`
   - Don't animate manually
   - Let DragAndScale handle it

4. **Canvas Lifecycle:** Component mounts → Get canvas ref → Start rendering
   - Don't render before canvas ready
   - Clean up animation frame on unmount

---

## Dependencies

**Required:**
- Vue 3 (already installed)
- TypeScript (already configured)
- Canvas 2D API (browser native)

**Optional:**
- d3-ease (for advanced animations) - NOT required
- PIXI.js (for WebGL rendering) - NOT required for basic version

**Not Needed:**
- Three.js (too heavy)
- Babylon.js (too heavy)
- Custom canvas frameworks

---

## Architecture Decision: Why Separate Component?

| Aspect | Separate Component | Hook-Based |
|--------|-------------------|-----------|
| Code clarity | High | Medium |
| Maintainability | High | Low |
| Vue integration | Native | Awkward |
| Testing | Easy | Hard |
| Reusability | Good | None |
| Performance | Same | Same |

**Verdict:** Use Vue component for GraphMinimap.

---

## Integration Steps (TL;DR)

1. Create GraphMinimap.vue with canvas
2. Add to GraphCanvas.vue template
3. Pass canvas instance via ref
4. Start rendering in onMounted
5. Listen for graph changes
6. Test with sample graphs

**Time:** 2-3 hours for Phase 1

---

## What NOT to Do

- ❌ Don't modify LGraphCanvas core
- ❌ Don't add minimap rendering to main draw loop
- ❌ Don't try to reuse main canvas for minimap
- ❌ Don't use complex node rendering (just rectangles)
- ❌ Don't forget to cleanup animation frame
- ❌ Don't animate every frame (only on changes)

---

## What TO Do

- ✓ Create new Vue component
- ✓ Use separate canvas element
- ✓ Access data via canvas reference
- ✓ Render in requestAnimationFrame
- ✓ Update only on graph changes
- ✓ Handle coordinate transforms carefully
- ✓ Clean up on component unmount
- ✓ Add TypeScript types for everything

---

## Configuration Template

```typescript
// In flowforge.config.ts
export const MINIMAP_CONFIG = {
  enabled: true,
  position: 'bottom-right',  // 'bottom-left', 'top-right', 'top-left'
  width: 200,
  height: 150,
  padding: 5,
  updateFrequency: 30,       // FPS
  colors: {
    background: '#1a1a1a',
    border: '#333',
    node: '#666',
    nodeHover: '#888',
    viewport: '#fff',
    viewportAlpha: 0.1
  },
  showHeader: true,
  collapsible: true,
  defaultCollapsed: false
}
```

---

## Component Skeleton

```vue
<template>
  <div class="minimap" v-if="!collapsed">
    <canvas
      ref="canvas"
      :width="width"
      :height="height"
      @click="handleClick"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import type { LGraphCanvas } from '@/litegraph'

const canvas = ref<HTMLCanvasElement | null>(null)
const width = ref(200)
const height = ref(150)
let lgraphCanvas: LGraphCanvas | null = null
let animId: number | null = null

function setCanvasInstance(c: LGraphCanvas) {
  lgraphCanvas = c
  startRendering()
}

function startRendering() {
  const render = () => {
    if (canvas.value && lgraphCanvas) {
      renderMinimap()
    }
    animId = requestAnimationFrame(render)
  }
  animId = requestAnimationFrame(render)
}

function renderMinimap() {
  // Implementation here
}

function handleClick(e: MouseEvent) {
  // Pan to clicked location
}

onUnmounted(() => {
  if (animId) cancelAnimationFrame(animId)
})

defineExpose({ setCanvasInstance })
</script>

<style scoped>
.minimap {
  position: absolute;
  bottom: 20px;
  right: 20px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 4px;
}

canvas {
  display: block;
  cursor: grab;
}

canvas:active {
  cursor: grabbing;
}
</style>
```

---

## Success Criteria

**Phase 1 Complete When:**
1. Minimap component renders nodes
2. Viewport rectangle tracks pan/zoom
3. Click-to-pan works smoothly
4. Integrated into GraphCanvas.vue
5. No performance issues
6. Code reviewed and documented

**Full Feature Complete When:**
- Phase 1 + Phase 2 + Phase 3 done
- 90%+ test coverage
- User feedback positive
- Production deployed

---

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Nodes not visible | Check coordinate transform math |
| Viewport box wrong | Verify `visible_area` calculation |
| Clicks don't pan | Debug click event handler |
| Lag/stutter | Reduce update frequency or node rendering |
| Canvas empty | Verify canvas instance passed correctly |
| Memory leak | Check animation frame cleanup on unmount |

---

## Performance Targets

- Minimap render: <5ms per frame
- Memory overhead: <100KB
- DOM elements: 1 (just canvas)
- No impact on main canvas FPS

---

## Related Documentation

- Full guide: `MINIMAP_IMPLEMENTATION_GUIDE.md`
- Research details: `RESEARCH_MINIMAP_3.2.5.md`
- Index: `RESEARCH_INDEX_3.2.5.md`
- Executive summary: `RESEARCH_SUMMARY_3.2.5.txt`

---

## Next Action

1. Read `MINIMAP_IMPLEMENTATION_GUIDE.md` (has full code)
2. Create GraphMinimap.vue with skeleton above
3. Implement rendering logic
4. Test with graph
5. Add to GraphCanvas.vue

**Estimated Time:** 2-3 hours to working prototype

---

**Status:** Ready to implement
**Difficulty:** Medium
**Confidence:** Very High

Go forth and minimap!
