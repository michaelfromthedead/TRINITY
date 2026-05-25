# ComfyUI Frontend Fork Plan

**Document:** Phase 1.2 Research - ComfyUI_frontend Repository Analysis
**Date:** 2026-01-27
**Repository:** https://github.com/Comfy-Org/ComfyUI_frontend

---

## Executive Summary

ComfyUI_frontend is a Vue 3 monorepo with a custom TypeScript fork of LiteGraph.js. The codebase is well-organized with clear separation between generic UI components and ComfyUI-specific functionality. The LiteGraph implementation is self-contained in `src/lib/litegraph/` making it ideal for extraction.

**Key Finding:** LiteGraph is NOT an npm dependency - it's a full TypeScript fork bundled directly in the source tree at `src/lib/litegraph/`. This is good news as we can copy it directly.

---

## 1. Repository Structure Overview

```
ComfyUI_frontend/
├── src/
│   ├── lib/litegraph/        # [KEEP] Full LiteGraph TypeScript implementation
│   ├── components/           # [PARTIAL] Vue UI components
│   ├── stores/               # [PARTIAL] Pinia state management
│   ├── services/             # [PARTIAL] Service layer
│   ├── composables/          # [PARTIAL] Vue composables
│   ├── scripts/              # [REVIEW] Legacy scripts including api.ts
│   ├── core/                 # [KEEP] Core graph schemas
│   ├── config/               # [KEEP] Configuration
│   ├── utils/                # [KEEP] Utilities
│   ├── views/                # [PARTIAL] Page components
│   ├── workbench/            # [REVIEW] Workbench features
│   ├── extensions/           # [REMOVE] ComfyUI extensions
│   ├── locales/              # [KEEP] i18n translations
│   ├── platform/             # [KEEP] Platform abstractions
│   └── renderer/             # [KEEP] Rendering logic
├── packages/
│   ├── design-system/        # [KEEP] UI components
│   ├── shared-frontend-utils/# [KEEP] Shared utilities
│   ├── tailwind-utils/       # [KEEP] Tailwind utilities
│   └── registry-types/       # [REMOVE] ComfyUI registry
└── apps/desktop-ui/          # [REFERENCE] Electron app (we use Tauri instead)
```

---

## 2. LiteGraph Analysis

### 2.1 Source Location
**Path:** `src/lib/litegraph/`

### 2.2 Core Files to Copy

| File | Purpose | Priority |
|------|---------|----------|
| `src/LGraph.ts` | Main graph container | Critical |
| `src/LGraphCanvas.ts` | Canvas rendering & interaction | Critical |
| `src/LGraphNode.ts` | Node implementation | Critical |
| `src/LLink.ts` | Connection links | Critical |
| `src/LGraphGroup.ts` | Node grouping | Critical |
| `src/CanvasPointer.ts` | Input handling | Critical |
| `src/DragAndScale.ts` | Pan/zoom operations | Critical |
| `src/ContextMenu.ts` | Right-click menus | Critical |
| `src/LGraphButton.ts` | UI buttons | High |
| `src/LGraphIcon.ts` | Icons | High |
| `src/LGraphBadge.ts` | Status indicators | High |
| `src/CurveEditor.ts` | Bezier curves | Medium |
| `src/Reroute.ts` | Connection routing | Medium |
| `src/constants.ts` | Constants | Critical |
| `src/draw.ts` | Drawing utilities | Critical |
| `src/measure.ts` | Size calculations | Critical |
| `src/LiteGraphGlobal.ts` | Global init | Critical |
| `src/MapProxyHandler.ts` | Data utilities | High |

### 2.3 Supporting Directories to Copy
- `src/lib/litegraph/src/` - All source files
- `src/lib/litegraph/public/css/` - Stylesheets
- `src/lib/litegraph/imgs/` - Image assets (if needed)

### 2.4 Copy Command
```bash
# From ComfyUI_frontend root
cp -r src/lib/litegraph/* /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/litegraph/
```

---

## 3. Vue Components Classification

### 3.1 KEEP - Generic UI Components

Copy to `apps/desktop/src/components/`:

| Component | Purpose |
|-----------|---------|
| `actionbar/` | Action control panels |
| `button/` | Button elements |
| `card/` | Card layouts |
| `chip/` | Chip/tag components |
| `dialog/` | Modal dialogs |
| `input/` | Form inputs |
| `tab/` | Tabbed interfaces |
| `toast/` | Notifications |
| `topbar/` | Top navigation |
| `bottomPanel/` | Bottom panels |
| `rightSidePanel/` | Side panels |
| `breadcrumb/` | Navigation |
| `searchbox/` | Search input |
| `ui/` | General UI utilities |
| `icons/` | Icon components |
| `common/` | Shared logic |

### 3.2 KEEP with Modifications - Graph Components

| Component | Purpose | Modifications Needed |
|-----------|---------|---------------------|
| `graph/` | Node graph editor | Remove SD-specific node types |
| `node/` | Node rendering | Generalize for any node type |
| `widget/` | Node widgets | Keep generic widgets only |
| `sidebar/` | Navigation sidebar | Remove SD model lists |

### 3.3 REMOVE - Stable Diffusion Specific

| Component | Reason |
|-----------|--------|
| `queue/` | SD job queue management |
| `maskeditor/` | Image mask editing |
| `imagecrop/` | Image cropping |
| `load3d/` | 3D model loading |
| `templates/thumbnails/` | SD template previews |
| `custom/widget/` | SD-specific widgets |
| `honeyToast/` | Custom notification (optional) |

---

## 4. Pinia Stores Classification

### 4.1 KEEP - Generic UI Stores

Copy to `apps/desktop/src/stores/`:

| Store | Purpose |
|-------|---------|
| `dialogStore.ts` | Modal management |
| `keybindingStore.ts` | Keyboard shortcuts |
| `menuItemStore.ts` | Menu state |
| `widgetStore.ts` | Widget state |
| `commandStore.ts` | Command palette |
| `extensionStore.ts` | Extension loading |
| `userStore.ts` | User preferences |
| `subgraphStore.ts` | Subgraph management |
| `subgraphNavigationStore.ts` | Navigation state |
| `nodeBookmarkStore.ts` | Node bookmarks |

**Workspace stores (keep all):**
| Store | Purpose |
|-------|---------|
| `bottomPanelStore.ts` | Bottom panel state |
| `colorPaletteStore.ts` | Color selection |
| `nodeHelpStore.ts` | Help content |
| `searchBoxStore.ts` | Search state |
| `sidebarTabStore.ts` | Sidebar tabs |

### 4.2 MODIFY - Stores Needing Changes

| Store | Current Purpose | Modification |
|-------|-----------------|--------------|
| `nodeDefStore.ts` | SD node definitions | Generalize for Trinity nodes |
| `workflowStore.ts` | SD workflows | Generalize for Python AST |
| `executionStore.ts` | SD execution | Replace with Python execution |
| `queueStore.ts` | SD queue | Replace with Python task queue |

### 4.3 REMOVE - SD-Specific Stores

| Store | Reason |
|-------|--------|
| `modelStore.ts` | SD model management |
| `modelToNodeStore.ts` | SD model-node mapping |
| `imagePreviewStore.ts` | SD image previews |
| `assetDownloadStore.ts` | SD asset downloads |
| `assetsStore.ts` | SD assets |
| `templateRankingStore.ts` | SD templates |
| `systemStatsStore.ts` | SD system stats |
| `maskEditorStore.ts` | SD mask editing |
| `maskEditorDataStore.ts` | SD mask data |
| `serverConfigStore.ts` | ComfyUI server config |
| `comfyRegistryStore.ts` | ComfyUI registry |
| `apiKeyAuthStore.ts` | ComfyUI API auth |
| `firebaseAuthStore.ts` | Firebase auth |
| `workspaceAuthStore.ts` | Workspace auth |
| `electronDownloadStore.ts` | Electron downloads |
| `helpCenterStore.ts` | ComfyUI help |
| `topbarBadgeStore.ts` | ComfyUI badges |
| `actionBarButtonStore.ts` | ComfyUI action bar |
| `aboutPanelStore.ts` | ComfyUI about |
| `domWidgetStore.ts` | DOM widgets |
| `userFileStore.ts` | User files |
| `bootstrapStore.ts` | ComfyUI bootstrap |

---

## 5. API Endpoints to Replace

### 5.1 Current ComfyUI API (from `api.ts`)

**GET Endpoints (Replace with Tauri IPC):**

| Endpoint | Purpose | FlowForge Replacement |
|----------|---------|----------------------|
| `/object_info` | Node definitions | `invoke('get_node_definitions')` |
| `/extensions` | Extensions list | `invoke('get_extensions')` |
| `/embeddings` | Embeddings | N/A (remove) |
| `/system_stats` | System stats | `invoke('get_system_info')` |
| `/users` | User config | Local storage |
| `/settings` | Settings | Local storage via Tauri |
| `/userdata/{file}` | User files | Tauri file system |
| `/logs` | Logs | Tauri logging |
| `/workflow_templates` | Templates | Local file system |
| `/experiment/models/*` | SD models | N/A (remove) |
| `/view_metadata/*` | Model metadata | N/A (remove) |
| `/i18n` | i18n data | Local bundles |
| `/global_subgraphs` | Subgraphs | Python sidecar |

**POST Endpoints (Replace with Tauri IPC):**

| Endpoint | Purpose | FlowForge Replacement |
|----------|---------|----------------------|
| `/prompt` | Execute workflow | `invoke('execute_workflow')` |
| `/interrupt` | Stop execution | `invoke('interrupt_execution')` |
| `/free` | Free memory | N/A (Python manages this) |
| `/settings` | Save settings | Local storage via Tauri |
| `/userdata/{file}` | Save files | Tauri file system |

### 5.2 WebSocket Events to Replace

**Execution Events (Replace with Tauri Events):**

| Event | Purpose | FlowForge Replacement |
|-------|---------|----------------------|
| `status` | System status | `listen('execution:status')` |
| `executing` | Node progress | `listen('execution:node')` |
| `execution_start` | Started | `listen('execution:start')` |
| `execution_success` | Complete | `listen('execution:success')` |
| `execution_error` | Error | `listen('execution:error')` |
| `progress` | Progress data | `listen('execution:progress')` |
| `progress_text` | Text updates | `listen('execution:progress_text')` |

**Binary Events (May not be needed):**

| Event | Purpose | FlowForge Action |
|-------|---------|-----------------|
| `b_preview` | Image preview | Remove (no image gen) |
| `b_preview_with_metadata` | Preview + meta | Remove |

---

## 6. Composables Classification

### 6.1 KEEP - Generic Utilities

Copy to `apps/desktop/src/composables/`:

| Composable | Purpose |
|------------|---------|
| `useBrowserTabTitle.ts` | Tab title |
| `useCachedRequest.ts` | Request caching |
| `useCopy.ts` | Copy operations |
| `useCopyToClipboard.ts` | Clipboard |
| `useDownload.ts` | File downloads |
| `useErrorHandling.ts` | Error handling |
| `useExternalLink.ts` | External links |
| `useIntersectionObserver.ts` | Visibility |
| `useLazyPagination.ts` | Pagination |
| `useRefreshableSelection.ts` | Selection state |
| `useTreeExpansion.ts` | Tree state |
| `useValueTransform.ts` | Value formatting |
| `useZoomControls.ts` | Zoom controls |
| `functional/` | Utility helpers |
| `element/` | DOM interactions |
| `tree/` | Tree structures |

### 6.2 KEEP with Modifications

| Composable | Modifications |
|------------|---------------|
| `canvas/` | Keep, update for Trinity |
| `graph/` | Keep, generalize |
| `node/` | Keep, generalize |

### 6.3 REMOVE

| Composable | Reason |
|------------|--------|
| `useCivitaiModel.ts` | SD-specific |
| `useModelSelectorDialog.ts` | SD models |
| `useTemplateFiltering.ts` | SD templates |
| `useProgressFavicon.ts` | SD progress |
| `useProgressBarBackground.ts` | SD progress |
| `maskeditor/` | SD mask editing |
| `queue/` | SD queue |
| `auth/` | ComfyUI auth |
| `bottomPanelTabs/` | Review (terminal may be useful) |
| `sidebarTabs/` | Review (keep if generic) |

---

## 7. Services Layer

### 7.1 KEEP

| Service | Purpose |
|---------|---------|
| `colorPaletteService.ts` | Color management |
| `dialogService.ts` | Dialog utilities |
| `keybindingService.ts` | Keyboard shortcuts |
| `litegraphService.ts` | LiteGraph bridge |
| `nodeSearchService.ts` | Node search |
| `nodeOrganizationService.ts` | Node organization |
| `subgraphService.ts` | Subgraph management |
| `nodeHelpService.ts` | Node help |

### 7.2 REMOVE

| Service | Reason |
|---------|--------|
| `audioService.ts` | SD audio |
| `autoQueueService.ts` | SD auto-queue |
| `comfyRegistryService.ts` | ComfyUI registry |
| `customerEventsService.ts` | Analytics |
| `extensionService.ts` | ComfyUI extensions |
| `jobOutputCache.ts` | SD job cache |
| `load3dService.ts` | 3D loading |
| `mediaCacheService.ts` | Media cache |
| `newUserService.ts` | User onboarding |
| `gateway/` | ComfyUI gateway |
| `providers/` | Search providers |

---

## 8. Scripts Layer

### 8.1 Analysis

**`src/scripts/api.ts`** - The main API client. This file contains:
- ComfyApi class with all fetch calls
- WebSocket connection management
- All endpoint definitions

**Action:** Create new `apps/desktop/src/bridge/tauri-api.ts` to replace this entirely.

### 8.2 KEEP with Modifications

| Script | Purpose | Modifications |
|--------|---------|---------------|
| `app.ts` | Core app management | Adapt for Tauri |
| `ui.ts` | UI system | Keep as-is |
| `widgets.ts` | Widget management | Keep as-is |
| `utils.ts` | Utilities | Keep as-is |
| `changeTracker.ts` | Change tracking | Keep as-is |
| `domWidget.ts` | DOM widgets | Keep as-is |
| `defaultGraph.ts` | Default graph | Update for Trinity |

### 8.3 REMOVE

| Script | Reason |
|--------|--------|
| `api.ts` | Replace with Tauri IPC |
| `pnginfo.ts` | SD PNG metadata |
| `errorNodeWidgets.ts` | SD-specific |
| `ui/` subdirectory | Review, may be legacy |

---

## 9. Implementation Plan

### Phase 1: Copy LiteGraph (Task 1.2.2)

```bash
# 1. Clone ComfyUI_frontend
git clone https://github.com/Comfy-Org/ComfyUI_frontend.git /tmp/comfyui-frontend
cd /tmp/comfyui-frontend

# 2. Copy LiteGraph
cp -r src/lib/litegraph/* /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/litegraph/

# 3. Copy CSS
mkdir -p /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/assets/litegraph
cp -r src/lib/litegraph/public/css/* /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/assets/litegraph/
```

### Phase 2: Copy Components (Task 1.2.3)

```bash
# Generic UI components
mkdir -p /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/components

# Copy each generic component
for dir in actionbar button card chip dialog input tab toast topbar \
           bottomPanel rightSidePanel breadcrumb searchbox ui icons common; do
    cp -r src/components/$dir /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/components/
done

# Copy graph components (need modification later)
for dir in graph node widget sidebar; do
    cp -r src/components/$dir /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/components/
done
```

### Phase 3: Copy Stores (Task 1.2.4)

```bash
# Generic stores
cd src/stores
for store in dialogStore keybindingStore menuItemStore widgetStore \
             commandStore extensionStore userStore subgraphStore \
             subgraphNavigationStore nodeBookmarkStore nodeDefStore \
             workflowStore executionStore queueStore; do
    cp ${store}.ts /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/stores/
done

# Workspace stores
cp -r workspace/* /home/user/dev/AI_GAME_ENGINE/flowforge/apps/desktop/src/stores/workspace/
```

### Phase 4: Remove SD-Specific Code (Task 1.2.5)

After copying, remove/comment out:

1. **In nodeDefStore.ts:**
   - SD node type definitions
   - Model/checkpoint/LoRA references

2. **In executionStore.ts:**
   - Image generation progress
   - SD-specific status handling

3. **In workflowStore.ts:**
   - SD workflow format
   - Image output handling

4. **In graph components:**
   - SD node previews
   - Model selector widgets
   - Image preview nodes

### Phase 5: Strip API Calls (Task 1.2.6)

Create `apps/desktop/src/bridge/tauri-api.ts`:

```typescript
import { invoke, listen } from '@tauri-apps/api';

export class FlowForgeApi {
  // Replace ComfyUI endpoints with Tauri IPC

  async getNodeDefinitions(): Promise<NodeDefinition[]> {
    return await invoke('get_node_definitions');
  }

  async executeWorkflow(workflow: Workflow): Promise<string> {
    return await invoke('execute_workflow', { workflow });
  }

  async interruptExecution(): Promise<void> {
    return await invoke('interrupt_execution');
  }

  // Event listeners
  onExecutionProgress(callback: (progress: Progress) => void): UnlistenFn {
    return listen('execution:progress', (event) => callback(event.payload));
  }

  onExecutionComplete(callback: (result: Result) => void): UnlistenFn {
    return listen('execution:complete', (event) => callback(event.payload));
  }
}

export const api = new FlowForgeApi();
```

---

## 10. File Copy Summary

### Files to Copy to `apps/desktop/src/litegraph/`

```
src/lib/litegraph/
├── src/
│   ├── LGraph.ts
│   ├── LGraphCanvas.ts
│   ├── LGraphNode.ts
│   ├── LLink.ts
│   ├── LGraphGroup.ts
│   ├── CanvasPointer.ts
│   ├── DragAndScale.ts
│   ├── ContextMenu.ts
│   ├── LGraphButton.ts
│   ├── LGraphIcon.ts
│   ├── LGraphBadge.ts
│   ├── CurveEditor.ts
│   ├── Reroute.ts
│   ├── constants.ts
│   ├── draw.ts
│   ├── measure.ts
│   ├── LiteGraphGlobal.ts
│   ├── MapProxyHandler.ts
│   └── [all other .ts files]
├── public/css/
└── package.json (for reference)
```

### Files to Copy to `apps/desktop/src/components/`

```
src/components/
├── actionbar/
├── button/
├── card/
├── chip/
├── dialog/
├── input/
├── tab/
├── toast/
├── topbar/
├── bottomPanel/
├── rightSidePanel/
├── breadcrumb/
├── searchbox/
├── ui/
├── icons/
├── common/
├── graph/          # Needs SD removal
├── node/           # Needs SD removal
├── widget/         # Needs SD removal
└── sidebar/        # Needs SD removal
```

### Files to Copy to `apps/desktop/src/stores/`

```
src/stores/
├── dialogStore.ts
├── keybindingStore.ts
├── menuItemStore.ts
├── widgetStore.ts
├── commandStore.ts
├── extensionStore.ts
├── userStore.ts
├── subgraphStore.ts
├── subgraphNavigationStore.ts
├── nodeBookmarkStore.ts
├── nodeDefStore.ts         # Needs modification
├── workflowStore.ts        # Needs modification
├── executionStore.ts       # Needs modification
├── queueStore.ts           # Needs modification
└── workspace/
    ├── bottomPanelStore.ts
    ├── colorPaletteStore.ts
    ├── nodeHelpStore.ts
    ├── searchBoxStore.ts
    └── sidebarTabStore.ts
```

### Files/Code to REMOVE (SD-Specific)

```
# Components to NOT copy
src/components/queue/
src/components/maskeditor/
src/components/imagecrop/
src/components/load3d/
src/components/templates/
src/components/custom/widget/

# Stores to NOT copy
src/stores/modelStore.ts
src/stores/modelToNodeStore.ts
src/stores/imagePreviewStore.ts
src/stores/assetDownloadStore.ts
src/stores/assetsStore.ts
src/stores/templateRankingStore.ts
src/stores/systemStatsStore.ts
src/stores/maskEditorStore.ts
src/stores/maskEditorDataStore.ts
src/stores/serverConfigStore.ts
src/stores/comfyRegistryStore.ts
src/stores/apiKeyAuthStore.ts
src/stores/firebaseAuthStore.ts
src/stores/workspaceAuthStore.ts
src/stores/electronDownloadStore.ts
src/stores/helpCenterStore.ts
src/stores/topbarBadgeStore.ts
src/stores/actionBarButtonStore.ts
src/stores/aboutPanelStore.ts
src/stores/domWidgetStore.ts
src/stores/userFileStore.ts
src/stores/bootstrapStore.ts

# Services to NOT copy
src/services/audioService.ts
src/services/autoQueueService.ts
src/services/comfyRegistryService.ts
src/services/customerEventsService.ts
src/services/extensionService.ts
src/services/jobOutputCache.ts
src/services/load3dService.ts
src/services/mediaCacheService.ts
src/services/newUserService.ts
src/services/gateway/
src/services/providers/

# Scripts to NOT copy
src/scripts/api.ts         # Replace with tauri-api.ts
src/scripts/pnginfo.ts     # SD-specific

# Composables to NOT copy
src/composables/useCivitaiModel.ts
src/composables/useModelSelectorDialog.ts
src/composables/useTemplateFiltering.ts
src/composables/useProgressFavicon.ts
src/composables/useProgressBarBackground.ts
src/composables/maskeditor/
src/composables/queue/
src/composables/auth/
```

---

## 11. Dependencies to Add

Based on ComfyUI_frontend's package.json:

```json
{
  "dependencies": {
    "pinia": "^2.x",
    "vue": "^3.x",
    "vue-router": "^4.x",
    "vue-i18n": "^9.x",
    "@vueuse/core": "^10.x",
    "primevue": "^4.x",
    "tailwindcss": "^3.x",
    "fuse.js": "^7.x",
    "jsonata": "^2.x"
  }
}
```

Note: Three.js and related 3D libraries are NOT needed for FlowForge.

---

## 12. Post-Copy Checklist

- [ ] LiteGraph compiles without errors
- [ ] Vue components have no missing imports
- [ ] Pinia stores initialize correctly
- [ ] No references to `/api/` endpoints remain
- [ ] No references to ComfyUI Python server remain
- [ ] No SD-specific node types in codebase
- [ ] Tauri IPC bridge created and working
- [ ] Canvas renders empty graph
- [ ] Context menu opens on right-click
- [ ] Nodes can be added from palette
- [ ] Nodes can be connected with links
- [ ] Pan and zoom work correctly

---

## 13. References

- ComfyUI_frontend: https://github.com/Comfy-Org/ComfyUI_frontend
- LiteGraph.js original: https://github.com/jagenjo/litegraph.js
- Tauri IPC documentation: https://v2.tauri.app/learn/calling-rust/
- FlowForge ADR-001: `/home/user/dev/AI_GAME_ENGINE/flowforge/docs/adr/001-monorepo-structure.md`
