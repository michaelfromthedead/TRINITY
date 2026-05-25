<template>
  <div
    :class="cn(
      'nav-title flex items-center gap-2 py-2 text-xs font-medium uppercase tracking-wide text-muted-foreground',
      collapsible && 'cursor-pointer hover:text-base-foreground'
    )"
    @click="toggleCollapse"
  >
    <i
      v-if="collapsible"
      :class="cn(
        'icon-[lucide--chevron-right] transition-transform',
        !modelValue && 'rotate-90'
      )"
    />
    <span>{{ title }}</span>
  </div>
</template>

<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'

const props = defineProps<{
  title: string
  collapsible?: boolean
  modelValue?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
}>()

const toggleCollapse = () => {
  if (props.collapsible) {
    emit('update:modelValue', !props.modelValue)
  }
}
</script>
