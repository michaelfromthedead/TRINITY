<template>
  <Dialog
    v-for="item in dialogStack"
    :key="item.key"
    v-model:visible="item.visible"
    :class="['global-dialog', item.dialogClass]"
    v-bind="getDialogProps(item)"
    :aria-labelledby="item.key"
  >
    <template #header>
      <div v-if="!getDialogProps(item)?.headless">
        <component
          :is="item.headerComponent"
          v-if="item.headerComponent"
          v-bind="item.headerProps"
          :id="item.key"
        />
        <h3 v-else :id="item.key">
          {{ item.title || ' ' }}
        </h3>
      </div>
    </template>

    <component
      :is="item.component"
      v-bind="item.contentProps"
      :maximized="getDialogProps(item)?.maximized"
    />

    <template v-if="item.footerComponent" #footer>
      <component :is="item.footerComponent" v-bind="item.footerProps" />
    </template>
  </Dialog>
</template>

<script setup lang="ts">
import Dialog from 'primevue/dialog'
import type { Component } from 'vue'

interface DialogItem {
  key: string
  visible: boolean
  title?: string
  dialogClass?: string
  /** Legacy prop name */
  dialogProps?: Record<string, unknown>
  /** New prop name from dialogStore */
  dialogComponentProps?: Record<string, unknown>
  component: Component
  contentProps?: Record<string, unknown>
  headerComponent?: Component
  headerProps?: Record<string, unknown>
  footerComponent?: Component
  footerProps?: Record<string, unknown>
}

defineProps<{
  dialogStack: DialogItem[]
}>()

/**
 * Get dialog props, supporting both legacy dialogProps and new dialogComponentProps.
 */
function getDialogProps(item: DialogItem): Record<string, unknown> | undefined {
  return item.dialogComponentProps ?? item.dialogProps
}
</script>

<style>
.global-dialog .p-dialog-header {
  padding: 0.5rem;
  padding-bottom: 0;
}

.global-dialog .p-dialog-content {
  padding: 0.5rem;
  padding-top: 0;
}

@media (min-width: 1536px) {
  .global-dialog .p-dialog-header {
    padding: var(--p-dialog-header-padding);
    padding-bottom: 0;
  }

  .global-dialog .p-dialog-content {
    padding: var(--p-dialog-content-padding);
    padding-top: 0;
  }
}
</style>
