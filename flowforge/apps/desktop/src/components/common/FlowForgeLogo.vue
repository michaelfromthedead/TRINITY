<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 32 32"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    class="flowforge-logo"
  >
    <circle cx="8" cy="10" r="3" :fill="color" />
    <circle cx="8" cy="22" r="3" :fill="color" />
    <circle cx="24" cy="16" r="3" :fill="color" />
    <line x1="11" y1="10" x2="21" y2="16" :stroke="color" stroke-width="2" stroke-linecap="round" />
    <line x1="11" y1="22" x2="21" y2="16" :stroke="color" stroke-width="2" stroke-linecap="round" />
  </svg>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  size?: number
  color?: string
}>(), {
  size: 20,
  color: '#6366f1'
})
</script>
