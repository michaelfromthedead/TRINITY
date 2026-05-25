<template>
  <div
    class="search-filter-chip inline-flex items-center gap-1 rounded-full bg-secondary-background px-2 py-1 text-xs"
  >
    <span>{{ text }}</span>
    <span
      v-if="badge"
      :class="cn('rounded-full bg-primary-background px-1.5 py-0.5 text-[10px]', badgeClass)"
    >
      {{ badge }}
    </span>
    <button
      type="button"
      class="ml-1 rounded-full p-0.5 hover:bg-secondary-background-hover"
      @click="emit('remove')"
    >
      <i class="icon-[lucide--x] size-3" />
    </button>
  </div>
</template>

<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'

export interface SearchFilter {
  id: string
  text: string
  badge?: string
  badgeClass?: string
}

defineProps<{
  text: string
  badge?: string
  badgeClass?: string
}>()

const emit = defineEmits<{
  (e: 'remove'): void
}>()
</script>
