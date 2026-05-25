<template>
  <Button
    v-show="isDeletable"
    v-tooltip.top="{
      value: 'Delete',
      showDelay: 1000
    }"
    variant="muted-textonly"
    aria-label="Delete selected items"
    data-testid="delete-button"
    @click="handleDelete"
  >
    <i class="icon-[lucide--trash-2]" />
  </Button>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import Button from '@/components/ui/button/Button.vue'

interface Positionable {
  removable?: boolean
}

const props = defineProps<{
  selectedItems?: Positionable[]
}>()

const emit = defineEmits<{
  (e: 'delete'): void
}>()

const isDeletable = computed(() =>
  props.selectedItems?.some((x: Positionable) => x.removable !== false) ?? false
)

const handleDelete = () => {
  emit('delete')
}
</script>
