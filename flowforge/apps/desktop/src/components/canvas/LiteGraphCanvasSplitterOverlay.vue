<template>
  <div
    class="w-full h-full absolute top-0 left-0 z-999 pointer-events-none flex flex-col"
  >
    <slot name="workflow-tabs" />

    <div
      class="pointer-events-none flex flex-1 overflow-hidden"
      :class="{
        'flex-row': sidebarLocation === 'left',
        'flex-row-reverse': sidebarLocation === 'right'
      }"
    >
      <div class="side-toolbar-container">
        <slot name="side-toolbar" />
      </div>

      <Splitter
        :key="splitterRefreshKey"
        class="bg-transparent pointer-events-none border-none flex-1 overflow-hidden"
        :state-key="sidebarStateKey"
        state-storage="local"
        @resizestart="onResizestart"
      >
        <!-- First panel: sidebar when left, properties when right -->
        <SplitterPanel
          v-if="
            !focusMode && (sidebarLocation === 'left' || rightSidePanelVisible)
          "
          :class="
            sidebarLocation === 'left'
              ? cn(
                  'side-bar-panel bg-panel-bg pointer-events-auto',
                  sidebarPanelVisible && 'min-w-78'
                )
              : 'bg-panel-bg pointer-events-auto'
          "
          :min-size="sidebarLocation === 'left' ? 10 : 15"
          :size="20"
          :style="firstPanelStyle"
          :role="sidebarLocation === 'left' ? 'complementary' : undefined"
          :aria-label="
            sidebarLocation === 'left' ? 'Sidebar' : undefined
          "
        >
          <slot
            v-if="sidebarLocation === 'left' && sidebarPanelVisible"
            name="side-bar-panel"
          />
          <slot
            v-else-if="sidebarLocation === 'right'"
            name="right-side-panel"
          />
        </SplitterPanel>

        <!-- Main panel (always present) -->
        <SplitterPanel :size="80" class="flex flex-col">
          <slot name="topmenu" :sidebar-panel-visible />

          <Splitter
            class="bg-transparent pointer-events-none border-none splitter-overlay-bottom mr-1 mb-1 ml-1 flex-1"
            layout="vertical"
            :pt:gutter="
              cn(
                'rounded-tl-lg rounded-tr-lg ',
                !(bottomPanelVisible && !focusMode) && 'hidden'
              )
            "
            state-key="bottom-panel-splitter"
            state-storage="local"
            @resizestart="onResizestart"
          >
            <SplitterPanel class="graph-canvas-panel relative pointer-events-auto">
              <slot name="graph-canvas-panel" />
            </SplitterPanel>
            <SplitterPanel
              v-show="bottomPanelVisible && !focusMode"
              class="bottom-panel border border-panel-border max-w-full overflow-x-auto bg-panel-bg pointer-events-auto rounded-lg"
            >
              <slot name="bottom-panel" />
            </SplitterPanel>
          </Splitter>
        </SplitterPanel>

        <!-- Last panel: properties when left, sidebar when right -->
        <SplitterPanel
          v-if="
            !focusMode && (sidebarLocation === 'right' || rightSidePanelVisible)
          "
          :class="
            sidebarLocation === 'right'
              ? cn(
                  'side-bar-panel bg-panel-bg pointer-events-auto',
                  sidebarPanelVisible && 'min-w-78'
                )
              : 'bg-panel-bg pointer-events-auto'
          "
          :min-size="sidebarLocation === 'right' ? 10 : 15"
          :size="20"
          :style="lastPanelStyle"
          :role="sidebarLocation === 'right' ? 'complementary' : undefined"
          :aria-label="
            sidebarLocation === 'right' ? 'Sidebar' : undefined
          "
        >
          <slot v-if="sidebarLocation === 'left'" name="right-side-panel" />
          <slot
            v-else-if="sidebarLocation === 'right' && sidebarPanelVisible"
            name="side-bar-panel"
          />
        </SplitterPanel>
      </Splitter>
    </div>
  </div>
</template>

<script setup lang="ts">
import Splitter from 'primevue/splitter'
import type { SplitterResizeStartEvent } from 'primevue/splitter'
import SplitterPanel from 'primevue/splitterpanel'
import { computed, ref } from 'vue'

import { cn } from '@/utils/tailwindUtil'

// Props for external control
const props = withDefaults(defineProps<{
  sidebarLocation?: 'left' | 'right'
  focusMode?: boolean
  sidebarPanelVisible?: boolean
  rightSidePanelVisible?: boolean
  bottomPanelVisible?: boolean
  unifiedWidth?: boolean
  activeSidebarTabId?: string | null
}>(), {
  sidebarLocation: 'left',
  focusMode: false,
  sidebarPanelVisible: false,
  rightSidePanelVisible: false,
  bottomPanelVisible: false,
  unifiedWidth: false,
  activeSidebarTabId: null
})

const sidebarLocation = computed(() => props.sidebarLocation)
const focusMode = computed(() => props.focusMode)
const sidebarPanelVisible = computed(() => props.sidebarPanelVisible)
const rightSidePanelVisible = computed(() => props.rightSidePanelVisible)
const bottomPanelVisible = computed(() => props.bottomPanelVisible)

const sidebarStateKey = computed(() => {
  return props.unifiedWidth
    ? 'unified-sidebar'
    : (props.activeSidebarTabId ?? 'default-sidebar')
})

/**
 * Avoid triggering default behaviors during drag-and-drop, such as text selection.
 */
function onResizestart({ originalEvent: event }: SplitterResizeStartEvent) {
  event.preventDefault()
}

/*
 * Force refresh the splitter when right panel visibility or sidebar location changes
 * to recalculate the width and panel order
 */
const splitterRefreshKey = computed(() => {
  return `main-splitter${rightSidePanelVisible.value ? '-with-right-panel' : ''}-${sidebarLocation.value}`
})

const firstPanelStyle = computed(() => {
  if (sidebarLocation.value === 'left') {
    return { display: sidebarPanelVisible.value ? 'flex' : 'none' }
  }
  return undefined
})

const lastPanelStyle = computed(() => {
  if (sidebarLocation.value === 'right') {
    return { display: sidebarPanelVisible.value ? 'flex' : 'none' }
  }
  return undefined
})
</script>

<style scoped>
:deep(.p-splitter-gutter) {
  pointer-events: auto;
}

:deep(.p-splitter-gutter:hover),
:deep(.p-splitter-gutter[data-p-gutter-resizing='true']) {
  transition: background-color 0.2s ease 300ms;
  background-color: var(--p-primary-color);
}

/* Hide sidebar gutter when sidebar is not visible */
:deep(.side-bar-panel[style*='display: none'] + .p-splitter-gutter),
:deep(.p-splitter-gutter + .side-bar-panel[style*='display: none']) {
  display: none;
}

.splitter-overlay-bottom :deep(.p-splitter-gutter) {
  transform: translateY(5px);
}

</style>
