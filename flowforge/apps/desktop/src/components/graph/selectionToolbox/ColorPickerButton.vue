<template>
  <div class="relative">
    <Button
      v-tooltip.top="{
        value: currentColorName ?? 'No color',
        showDelay: 1000
      }"
      data-testid="color-picker-button"
      variant="muted-textonly"
      aria-label="Color"
      @click="() => (showColorPicker = !showColorPicker)"
    >
      <div class="flex items-center gap-1 px-0">
        <i class="pi pi-circle-fill" :style="{ color: currentColor ?? '' }" />
        <i class="icon-[lucide--chevron-down]" />
      </div>
    </Button>
    <div
      v-if="showColorPicker"
      class="color-picker-container absolute -top-10 left-1/2"
    >
      <SelectButton
        :model-value="selectedColorOption"
        :options="colorOptions"
        option-label="name"
        data-key="value"
        @update:model-value="applyColor"
      >
        <template #option="{ option }">
          <i
            v-tooltip.top="option.name"
            class="pi pi-circle-fill"
            :style="{
              color: isLightTheme ? option.value.light : option.value.dark
            }"
            :data-testid="option.name"
          />
        </template>
      </SelectButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import SelectButton from 'primevue/selectbutton'
import { computed, ref, watch } from 'vue'

import Button from '@/components/ui/button/Button.vue'

interface ColorOption {
  name: string
  value: {
    dark: string
    light: string
  }
}

interface CanvasColorOption {
  bgcolor: string
  color?: string
  groupcolor?: string
}

interface Colorable {
  setColorOption: (option: CanvasColorOption | null) => void
}

const props = defineProps<{
  selectedItems?: Array<{ setColorOption?: (option: CanvasColorOption | null) => void }>
  isLightTheme?: boolean
  nodeColors?: Record<string, { bgcolor: string }>
  defaultBgColor?: string
}>()

const emit = defineEmits<{
  (e: 'colorChange', colorOption: CanvasColorOption | null): void
}>()

const isLightTheme = computed(() => props.isLightTheme ?? false)
const showColorPicker = ref(false)

// Adjust color for light theme
const toLightThemeColor = (color: string) => {
  // Simple lightness adjustment - in production you'd use a proper color library
  return color
}

const NO_COLOR_OPTION: ColorOption = {
  name: 'No Color',
  value: {
    dark: props.defaultBgColor ?? '#333355',
    light: props.defaultBgColor ?? '#888888'
  }
}

const colorOptions = computed<ColorOption[]>(() => {
  const options: ColorOption[] = [NO_COLOR_OPTION]

  if (props.nodeColors) {
    Object.entries(props.nodeColors).forEach(([name, color]) => {
      options.push({
        name,
        value: {
          dark: color.bgcolor,
          light: toLightThemeColor(color.bgcolor)
        }
      })
    })
  }

  return options
})

const selectedColorOption = ref<ColorOption | null>(null)
const currentColorOption = ref<CanvasColorOption | null>(null)

const currentColor = computed(() =>
  currentColorOption.value
    ? isLightTheme.value
      ? toLightThemeColor(currentColorOption.value.bgcolor)
      : currentColorOption.value.bgcolor
    : null
)

const currentColorName = computed(() => {
  if (!currentColorOption.value?.bgcolor) return null
  const colorOption = colorOptions.value.find(
    (option) =>
      option.value.dark === currentColorOption.value?.bgcolor ||
      option.value.light === currentColorOption.value?.bgcolor
  )
  return colorOption?.name ?? NO_COLOR_OPTION.name
})

const isColorable = (item: unknown): item is Colorable => {
  return typeof item === 'object' && item !== null && 'setColorOption' in item
}

const applyColor = (colorOption: ColorOption | null) => {
  const colorName = colorOption?.name ?? NO_COLOR_OPTION.name
  const canvasColorOption: CanvasColorOption | null =
    colorName === NO_COLOR_OPTION.name
      ? null
      : props.nodeColors?.[colorName]
        ? { bgcolor: props.nodeColors[colorName].bgcolor }
        : null

  if (props.selectedItems) {
    for (const item of props.selectedItems) {
      if (isColorable(item)) {
        item.setColorOption(canvasColorOption)
      }
    }
  }

  currentColorOption.value = canvasColorOption
  showColorPicker.value = false
  emit('colorChange', canvasColorOption)
}

// Update color from selected items
watch(
  () => props.selectedItems,
  () => {
    showColorPicker.value = false
    selectedColorOption.value = null
    // In a full implementation, you'd extract the current color from selected items
    currentColorOption.value = null
  },
  { immediate: true }
)
</script>

<style scoped>
.color-picker-container {
  transform: translateX(-50%);
}

:deep(.p-togglebutton) {
  padding: 0.5rem 0.25rem;
}
</style>
