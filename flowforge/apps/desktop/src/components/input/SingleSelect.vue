<template>
  <Select
    v-model="selectedItem"
    v-bind="$attrs"
    :options="options"
    option-label="name"
    option-value="value"
    unstyled
    :pt="selectPt"
    :aria-label="label || 'Select an option'"
    role="combobox"
    :aria-expanded="false"
    aria-haspopup="listbox"
    :tabindex="0"
  >
    <!-- Trigger value -->
    <template #value="slotProps">
      <div class="flex items-center gap-2 text-sm">
        <slot name="icon" />
        <span
          v-if="slotProps.value !== null && slotProps.value !== undefined"
          class="text-base-foreground"
        >
          {{ getLabel(slotProps.value) }}
        </span>
        <span v-else class="text-base-foreground">
          {{ label }}
        </span>
      </div>
    </template>

    <!-- Trigger caret -->
    <template #dropdownicon>
      <i class="icon-[lucide--chevron-down] text-muted-foreground" />
    </template>

    <!-- Option row -->
    <template #option="{ option, selected }">
      <div
        class="flex w-full items-center justify-between gap-3"
        :style="optionStyle"
      >
        <span class="truncate">{{ option.name }}</span>
        <i v-if="selected" class="icon-[lucide--check] text-base-foreground" />
      </div>
    </template>
  </Select>
</template>

<script setup lang="ts">
import Select from 'primevue/select'
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

export interface SelectOption {
  name: string
  value: string
}

defineOptions({
  inheritAttrs: false
})

const {
  label,
  options,
  listMaxHeight = '28rem',
  popoverMinWidth,
  popoverMaxWidth
} = defineProps<{
  label?: string
  options?: SelectOption[]
  listMaxHeight?: string
  popoverMinWidth?: string
  popoverMaxWidth?: string
}>()

const selectedItem = defineModel<string | undefined>({ required: true })

const getLabel = (val: string | null | undefined) => {
  if (val == null) return label ?? ''
  if (!options) return label ?? ''
  const found = options.find((o) => o.value === val)
  return found ? found.name : (label ?? '')
}

const optionStyle = computed(() => {
  if (!popoverMinWidth && !popoverMaxWidth) return undefined

  const styles: string[] = []
  if (popoverMinWidth) styles.push(`min-width: ${popoverMinWidth}`)
  if (popoverMaxWidth) styles.push(`max-width: ${popoverMaxWidth}`)

  return styles.join('; ')
})

// PrimeVue pass-through styling configuration
// Moved here to avoid vue-tsc template parsing issues with complex typed props
const selectPt = computed(() => ({
  root: (opts: { props: { disabled?: boolean } }) => ({
    class: [
      'h-10 relative inline-flex cursor-pointer select-none items-center',
      'rounded-lg',
      'bg-secondary-background text-base-foreground',
      'border-[2.5px] border-solid border-transparent',
      'transition-all duration-200 ease-in-out',
      'focus-within:border-primary-background',
      { 'opacity-60 cursor-default': opts.props.disabled }
    ]
  }),
  label: {
    class: 'flex-1 flex items-center whitespace-nowrap pl-4 py-2 outline-hidden'
  },
  dropdown: {
    class: 'flex shrink-0 items-center justify-center px-3 py-2'
  },
  overlay: {
    class: cn(
      'mt-2 p-2 rounded-lg',
      'bg-base-background text-base-foreground',
      'border border-solid border-border-default'
    )
  },
  listContainer: () => ({
    style: `max-height: min(${listMaxHeight}, 50vh)`,
    class: 'scrollbar-custom'
  }),
  list: {
    class: 'flex flex-col gap-0 p-0 m-0 list-none border-none text-sm'
  },
  option: (ctx: { context: { focused: boolean; selected: boolean } }) => ({
    class: cn(
      'flex items-center justify-between gap-3 px-2 py-3 rounded',
      'hover:bg-secondary-background-hover',
      ctx.context.focused && 'bg-secondary-background-hover',
      ctx.context.selected &&
        'bg-secondary-background-selected hover:bg-secondary-background-selected'
    )
  }),
  optionLabel: {
    class: 'truncate'
  },
  optionGroupLabel: {
    class: 'px-3 py-2 text-xs uppercase tracking-wide text-muted-foreground'
  },
  emptyMessage: {
    class: 'px-3 py-2 text-sm text-muted-foreground'
  }
}))
</script>
