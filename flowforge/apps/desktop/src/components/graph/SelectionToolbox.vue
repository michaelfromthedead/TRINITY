<template>
  <div
    ref="toolboxRef"
    style="transform: translate(var(--tb-x), var(--tb-y))"
    class="pointer-events-none fixed top-0 left-0 z-40"
  >
    <Transition name="slide-up">
      <Panel
        v-if="visible"
        class="selection-toolbox pointer-events-auto rounded-lg border border-interface-stroke bg-interface-panel-surface"
        :pt="{
          header: 'hidden',
          content: 'p-1 h-10 flex flex-row gap-1'
        }"
        @wheel="handleWheel"
      >
        <DeleteButton
          v-if="showDelete"
          :selected-items="selectedItems"
          @delete="emit('deleteItems')"
        />
        <VerticalDivider v-if="showDivider" />

        <ColorPickerButton
          v-if="showColorPicker"
          :selected-items="selectedItems"
          :is-light-theme="isLightTheme"
          :node-colors="nodeColors"
          :default-bg-color="defaultBgColor"
          @color-change="(color) => emit('colorChange', color)"
        />

        <slot name="extra-buttons" />
      </Panel>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import Panel from 'primevue/panel'
import { computed, ref } from 'vue'

import ColorPickerButton from './selectionToolbox/ColorPickerButton.vue'
import DeleteButton from './selectionToolbox/DeleteButton.vue'
import VerticalDivider from './selectionToolbox/VerticalDivider.vue'

interface Positionable {
  removable?: boolean
  setColorOption?: (option: unknown) => void
}

interface CanvasColorOption {
  bgcolor: string
  color?: string
  groupcolor?: string
}

const props = defineProps<{
  visible?: boolean
  selectedItems?: Positionable[]
  isLightTheme?: boolean
  nodeColors?: Record<string, { bgcolor: string }>
  defaultBgColor?: string
}>()

const emit = defineEmits<{
  (e: 'deleteItems'): void
  (e: 'colorChange', color: CanvasColorOption | null): void
  (e: 'wheel', event: WheelEvent): void
}>()

const toolboxRef = ref<HTMLElement | undefined>()

const visible = computed(() => props.visible ?? false)
const selectedItems = computed(() => props.selectedItems ?? [])

const hasAnySelection = computed(() => selectedItems.value.length > 0)

const showColorPicker = computed(() => hasAnySelection.value)
const showDelete = computed(() => hasAnySelection.value)
const showDivider = computed(() => showColorPicker.value && showDelete.value)

const handleWheel = (event: WheelEvent) => {
  emit('wheel', event)
}

defineExpose({
  toolboxRef
})
</script>

<style scoped>
.selection-toolbox {
  transform: translateX(-50%) translateY(-120%);
}

@keyframes slideUp {
  0% {
    transform: translateX(-50%) translateY(-100%);
    opacity: 0;
  }
  50% {
    transform: translateX(-50%) translateY(-125%);
    opacity: 0.5;
  }
  100% {
    transform: translateX(-50%) translateY(-120%);
    opacity: 1;
  }
}

.slide-up-enter-active {
  animation: slideUp 125ms ease-out;
}

.slide-up-leave-active {
  animation: slideUp 25ms ease-out reverse;
}
</style>
