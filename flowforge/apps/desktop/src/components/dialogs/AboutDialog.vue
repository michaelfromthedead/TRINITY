<template>
  <div class="about-dialog">
    <div class="about-header">
      <FlowForgeLogo :size="48" />
      <h2 class="about-title">FlowForge</h2>
      <span class="about-version">v{{ version }}</span>
    </div>

    <p class="about-description">Visual Python ECS Editor for Trinity</p>

    <div class="about-tech">
      <span class="tech-label">Built with</span>
      <div class="tech-list">
        <span class="tech-badge">Tauri</span>
        <span class="tech-badge">Vue</span>
        <span class="tech-badge">LiteGraph</span>
        <span class="tech-badge">Python</span>
      </div>
    </div>

    <div class="about-footer">
      <button type="button" class="btn btn-primary" @click="emit('cancel')">
        Close
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import FlowForgeLogo from '@/components/common/FlowForgeLogo.vue'

const version = '0.1.0'

const emit = defineEmits<{
  (e: 'cancel'): void
}>()
</script>

<style scoped>
.about-dialog {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  min-width: 320px;
  max-width: 400px;
  padding: 1.5rem 1rem 0.5rem;
  text-align: center;
}

.about-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.about-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--p-text-color, #fff);
}

.about-version {
  font-size: 0.8rem;
  color: var(--p-text-muted-color, #888);
}

.about-description {
  margin: 0;
  font-size: 0.875rem;
  color: var(--p-text-muted-color, #ccc);
}

.about-tech {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.tech-label {
  font-size: 0.75rem;
  color: var(--p-text-muted-color, #888);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.tech-list {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  justify-content: center;
}

.tech-badge {
  padding: 0.25rem 0.625rem;
  font-size: 0.75rem;
  color: var(--p-text-color, #ddd);
  background-color: var(--p-surface-200, #333);
  border-radius: 1rem;
}

.about-footer {
  display: flex;
  justify-content: center;
  padding-top: 0.75rem;
  border-top: 1px solid var(--p-surface-border, #3d3d3d);
  width: 100%;
}

.btn {
  padding: 0.5rem 1.5rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  border: none;
  transition: all 0.15s ease;
}

.btn-primary {
  background-color: #6366f1;
  color: white;
}

.btn-primary:hover {
  background-color: #4f46e5;
}
</style>
