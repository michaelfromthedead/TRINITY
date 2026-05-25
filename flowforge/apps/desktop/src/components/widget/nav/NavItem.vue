<template>
  <button
    type="button"
    :class="cn(
      'nav-item flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
      'hover:bg-secondary-background-hover',
      active && 'bg-secondary-background-selected text-primary-foreground'
    )"
    @click="emit('click')"
  >
    <i v-if="icon" :class="icon" />
    <span class="flex-1 text-left">
      <slot />
    </span>
    <span
      v-if="badge"
      class="rounded-full bg-primary-background px-2 py-0.5 text-xs text-base-foreground"
    >
      {{ badge }}
    </span>
  </button>
</template>

<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'

defineProps<{
  icon?: string
  badge?: string | number
  active?: boolean
}>()

const emit = defineEmits<{
  (e: 'click'): void
}>()
</script>
