<template>
  <div
    v-if="visible"
    class="absolute right-0 bottom-[62px] z-1300 flex w-[250px] justify-center border-0! bg-inherit!"
  >
    <div
      class="w-4/5 rounded-lg border border-interface-stroke bg-interface-panel-surface p-2 text-text-primary shadow-lg select-none"
      @click.stop
    >
      <div class="flex flex-col gap-1">
        <div
          class="flex cursor-pointer items-center justify-between rounded px-3 py-2 text-sm hover:bg-secondary-background-hover"
          @mousedown="startRepeat('zoomIn')"
          @mouseup="stopRepeat"
          @mouseleave="stopRepeat"
        >
          <span class="font-medium">Zoom In</span>
          <span class="text-[9px] text-text-primary">{{ zoomInShortcut }}</span>
        </div>

        <div
          class="flex cursor-pointer items-center justify-between rounded px-3 py-2 text-sm hover:bg-secondary-background-hover"
          @mousedown="startRepeat('zoomOut')"
          @mouseup="stopRepeat"
          @mouseleave="stopRepeat"
        >
          <span class="font-medium">Zoom Out</span>
          <span class="text-[9px] text-text-primary">{{ zoomOutShortcut }}</span>
        </div>

        <div
          class="flex cursor-pointer items-center justify-between rounded px-3 py-2 text-sm hover:bg-secondary-background-hover"
          @click="emit('fitView')"
        >
          <span class="font-medium">Zoom to Fit</span>
          <span class="text-[9px] text-text-primary">{{ fitViewShortcut }}</span>
        </div>

        <div
          ref="zoomInputContainer"
          class="zoomInputContainer flex items-center gap-1 rounded bg-input-surface p-2"
        >
          <InputNumber
            :default-value="zoomPercentage"
            :min="1"
            :max="1000"
            :show-buttons="false"
            :use-grouping="false"
            :unstyled="true"
            input-class="bg-transparent border-none outline-hidden text-sm shadow-none my-0 w-full"
            fluid
            @input="applyZoom"
            @keyup.enter="applyZoom"
          />
          <span class="flex-shrink-0 text-sm text-text-primary">%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { InputNumberInputEvent } from 'primevue'
import { InputNumber } from 'primevue'
import { nextTick, ref, watch } from 'vue'

const props = defineProps<{
  visible: boolean
  zoomPercentage?: number
  zoomInShortcut?: string
  zoomOutShortcut?: string
  fitViewShortcut?: string
}>()

const emit = defineEmits<{
  (e: 'zoomIn'): void
  (e: 'zoomOut'): void
  (e: 'fitView'): void
  (e: 'setZoom', percentage: number): void
}>()

const interval = ref<number | null>(null)

const applyZoom = (val: InputNumberInputEvent) => {
  const inputValue = val.value as number
  if (isNaN(inputValue) || inputValue < 1 || inputValue > 1000) {
    return
  }
  emit('setZoom', inputValue)
}

const startRepeat = (action: 'zoomIn' | 'zoomOut') => {
  if (interval.value) return
  const cmd = () => emit(action)
  cmd()
  interval.value = window.setInterval(cmd, 100)
}

const stopRepeat = () => {
  if (interval.value) {
    clearInterval(interval.value)
    interval.value = null
  }
}

const zoomInputContainer = ref<HTMLDivElement | null>(null)

watch(
  () => props.visible,
  async (newVal) => {
    if (newVal) {
      await nextTick()
      const input = zoomInputContainer.value?.querySelector(
        'input'
      ) as HTMLInputElement
      input?.focus()
    }
  }
)
</script>

<style>
.zoomInputContainer:focus-within {
  border: 1px solid var(--color-white);
}
</style>
