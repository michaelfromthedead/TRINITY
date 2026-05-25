<template>
  <Button
    v-tooltip="{
      value: computedTooltip,
      showDelay: 300,
      hideDelay: 300
    }"
    :class="
      cn(
        'side-bar-button cursor-pointer border-none',
        selected && 'side-bar-button-selected'
      )
    "
    variant="muted-textonly"
    :aria-label="computedTooltip"
    @click="emit('click', $event)"
  >
    <div class="side-bar-button-content">
      <slot name="icon">
        <div class="sidebar-icon-wrapper relative">
          <i
            v-if="typeof icon === 'string'"
            :class="icon + ' side-bar-button-icon'"
          />
          <component
            :is="icon"
            v-else-if="typeof icon === 'object'"
            class="side-bar-button-icon"
          />
          <span
            v-if="shouldShowBadge"
            :class="
              cn(
                'sidebar-icon-badge absolute min-w-[16px] rounded-full bg-primary-background py-0.25 text-[10px] font-medium leading-[14px] text-base-foreground',
                badgeClass || '-top-1 -right-1'
              )
            "
          >
            {{ overlayValue }}
          </span>
        </div>
      </slot>
      <span v-if="label && !isSmall" class="side-bar-button-label">{{
        label
      }}</span>
    </div>
  </Button>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Component } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { cn } from '@/utils/tailwindUtil'

const {
  icon = '',
  selected = false,
  tooltip = '',
  tooltipSuffix = '',
  iconBadge = '',
  badgeClass = '',
  label = '',
  isSmall = false
} = defineProps<{
  icon?: string | Component
  selected?: boolean
  tooltip?: string
  tooltipSuffix?: string
  iconBadge?: string | (() => string | null)
  badgeClass?: string
  label?: string
  isSmall?: boolean
}>()

const emit = defineEmits<{
  (e: 'click', event: MouseEvent): void
}>()

const overlayValue = computed(() =>
  typeof iconBadge === 'function' ? (iconBadge() ?? '') : iconBadge
)
const shouldShowBadge = computed(() => !!overlayValue.value)
const computedTooltip = computed(() => tooltip + tooltipSuffix)
</script>

<style>
.side-bar-button-icon {
  font-size: var(--sidebar-icon-size, 1rem) !important;
}

.side-bar-button-selected {
  background-color: var(--interface-panel-selected-surface);
  color: var(--content-hover-fg);
}
.side-bar-button:hover {
  background-color: var(--interface-panel-hover-surface);
  color: var(--content-hover-fg);
}

.side-bar-button-selected .side-bar-button-icon {
  font-size: var(--sidebar-icon-size, 1rem) !important;
}
</style>

<style scoped>
.side-bar-button {
  width: var(--sidebar-width, 48px);
  height: var(--sidebar-item-height, 56px);
  border-radius: 0;
  flex-shrink: 0;
}

.side-tool-bar-end .side-bar-button {
  height: var(--sidebar-width, 48px);
}

.side-bar-button-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.side-bar-button-label {
  font-size: 10px;
  text-align: center;
  line-height: 1;
}
</style>
