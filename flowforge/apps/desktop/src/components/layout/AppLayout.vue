<template>
  <div class="app-layout">
    <LiteGraphCanvasSplitterOverlay
      :sidebar-location="sidebarLocation"
      :focus-mode="focusMode"
      :sidebar-panel-visible="sidebarVisible"
      :right-side-panel-visible="rightPanelVisible"
      :bottom-panel-visible="bottomPanelVisible"
      :active-sidebar-tab-id="activeSidebarTabId"
    >
      <!-- Workflow tabs slot -->
      <template #workflow-tabs>
        <slot name="workflow-tabs" />
      </template>

      <!-- Side toolbar (icon bar) -->
      <template #side-toolbar>
        <div class="side-toolbar">
          <slot name="side-toolbar">
            <div class="side-toolbar-icons">
              <SidebarIcon
                v-for="tab in sidebarTabs"
                :key="tab.id"
                :icon="tab.icon"
                :tooltip="tab.title"
                :selected="activeSidebarTabId === tab.id"
                @click="toggleSidebarTab(tab.id)"
              />
            </div>
          </slot>
        </div>
      </template>

      <!-- Sidebar panel content -->
      <template #side-bar-panel>
        <div class="sidebar-panel">
          <slot name="sidebar">
            <!-- Default sidebar content -->
            <div class="sidebar-placeholder">
              <p>Sidebar content</p>
            </div>
          </slot>
        </div>
      </template>

      <!-- Top menu bar -->
      <template #topmenu="{ sidebarPanelVisible }">
        <slot name="topmenu" :sidebar-visible="sidebarPanelVisible" />
      </template>

      <!-- Main graph canvas area -->
      <template #graph-canvas-panel>
        <div class="graph-canvas-wrapper">
          <slot name="canvas">
            <!-- Default canvas placeholder -->
            <div class="canvas-placeholder">
              <p>Graph Canvas</p>
            </div>
          </slot>

          <!-- Canvas overlays (toolbox, zoom controls, etc.) -->
          <slot name="canvas-overlays" />
        </div>
      </template>

      <!-- Right side panel (properties, inspector) -->
      <template #right-side-panel>
        <div class="right-panel">
          <slot name="right-panel">
            <!-- Default right panel content -->
            <div class="panel-placeholder">
              <p>Properties</p>
            </div>
          </slot>
        </div>
      </template>

      <!-- Bottom panel (terminal, output, etc.) -->
      <template #bottom-panel>
        <div class="bottom-panel-content">
          <slot name="bottom-panel">
            <!-- Default bottom panel content -->
            <div class="panel-placeholder">
              <p>Output</p>
            </div>
          </slot>
        </div>
      </template>
    </LiteGraphCanvasSplitterOverlay>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import LiteGraphCanvasSplitterOverlay from '@/components/canvas/LiteGraphCanvasSplitterOverlay.vue'
import SidebarIcon from '@/components/sidebar/SidebarIcon.vue'
import { useSidebarTabStore } from '@/stores/sidebarTabStore'
import { useBottomPanelStore } from '@/stores/bottomPanelStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'

// =============================================================================
// PROPS
// =============================================================================

const props = withDefaults(defineProps<{
  sidebarLocation?: 'left' | 'right'
  showRightPanel?: boolean
}>(), {
  sidebarLocation: 'left',
  showRightPanel: false
})

// =============================================================================
// STORES
// =============================================================================

const sidebarTabStore = useSidebarTabStore()
const bottomPanelStore = useBottomPanelStore()
const workspaceStore = useWorkspaceStore()

// =============================================================================
// COMPUTED
// =============================================================================

const sidebarLocation = computed(() => props.sidebarLocation)
const focusMode = computed(() => workspaceStore.focusMode)
const sidebarTabs = computed(() => sidebarTabStore.sortedSidebarTabs)
const activeSidebarTabId = computed(() => sidebarTabStore.activeSidebarTabId)
const sidebarVisible = computed(() => !!activeSidebarTabId.value)
const bottomPanelVisible = computed(() => bottomPanelStore.bottomPanelVisible)

// Right panel visibility - controlled by prop
const rightPanelVisible = computed(() => props.showRightPanel)

// =============================================================================
// METHODS
// =============================================================================

function toggleSidebarTab(tabId: string) {
  sidebarTabStore.toggleSidebarTab(tabId)
}
</script>

<style scoped>
.app-layout {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background-color: var(--app-bg, #1e1e1e);
}

.side-toolbar {
  display: flex;
  flex-direction: column;
  background-color: var(--toolbar-bg, #2d2d2d);
  border-right: 1px solid var(--border-color, #3d3d3d);
  pointer-events: auto;
}

.side-toolbar-icons {
  display: flex;
  flex-direction: column;
}

.sidebar-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.graph-canvas-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.right-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  border-left: 1px solid var(--border-color, #3d3d3d);
}

.bottom-panel-content {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.sidebar-placeholder,
.canvas-placeholder,
.panel-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-muted, #666);
  font-size: 14px;
}

.canvas-placeholder {
  background-color: var(--canvas-bg, #1a1a2e);
}
</style>
