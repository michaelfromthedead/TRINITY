<template>
  <div class="inspector-panel">
    <!-- Header -->
    <div class="inspector-header">
      <h2 class="inspector-title">Inspector</h2>
    </div>

    <!-- Empty State -->
    <div v-if="!hasSelection" class="inspector-empty">
      <div class="empty-icon">
        <svg
          width="48"
          height="48"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="16" x2="12" y2="12" />
          <line x1="12" y1="8" x2="12.01" y2="8" />
        </svg>
      </div>
      <p class="empty-text">Select a node to inspect</p>
      <p class="empty-hint">Click on a node in the graph to view its details</p>
    </div>

    <!-- Loading State -->
    <div v-else-if="isLoading" class="inspector-loading">
      <div class="loading-spinner" />
      <p class="loading-text">Loading...</p>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="inspector-error">
      <div class="error-icon">
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="15" y1="9" x2="9" y2="15" />
          <line x1="9" y1="9" x2="15" y2="15" />
        </svg>
      </div>
      <p class="error-text">{{ error }}</p>
    </div>

    <!-- Inspector Content -->
    <div v-else-if="inspectorData" class="inspector-content">
      <!-- Class Header -->
      <div class="class-header">
        <div class="class-name-row">
          <span
            class="trinity-badge"
            :style="{ backgroundColor: getTrinityTypeColor(inspectorData.trinityType) }"
          >
            {{ inspectorData.trinityType }}
          </span>
          <h3 class="class-name">{{ inspectorData.className }}</h3>
        </div>
        <p class="module-name">{{ inspectorData.moduleName }}</p>
        <p v-if="inspectorData.docstring" class="docstring">{{ inspectorData.docstring }}</p>
      </div>

      <!-- Source Location -->
      <div class="source-location">
        <div class="source-info">
          <span class="source-label">Source:</span>
          <span class="source-path" :title="inspectorData.source.file">
            {{ formatSourcePath(inspectorData.source.file) }}
          </span>
          <span v-if="inspectorData.source.line" class="source-line">
            :{{ inspectorData.source.line }}
            <span v-if="inspectorData.source.endLine">-{{ inspectorData.source.endLine }}</span>
          </span>
        </div>
        <button class="open-editor-btn" @click="handleViewSource" :title="'Open in Editor: ' + inspectorData.source.file">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
            <polyline points="15 3 21 3 21 9" />
            <line x1="10" y1="14" x2="21" y2="3" />
          </svg>
          <span>Open in Editor</span>
        </button>
      </div>

      <!-- Collapsible Sections -->
      <div class="inspector-sections">
        <!-- Inheritance Section -->
        <InspectorSection
          title="Inheritance"
          :is-collapsed="sectionState.inheritance"
          :item-count="inspectorData.inheritance.bases.length"
          @toggle="toggleSection('inheritance')"
        >
          <div class="inheritance-chain">
            <div
              v-for="(base, index) in inspectorData.inheritance.bases"
              :key="index"
              class="base-class"
              :class="{ 'trinity-base': base.isTrinityBase }"
            >
              <span class="base-icon">
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                >
                  <polyline points="18,15 12,9 6,15" />
                </svg>
              </span>
              <span class="base-name">{{ base.name }}</span>
              <span v-if="base.module" class="base-module">({{ base.module }})</span>
            </div>
            <div v-if="inspectorData.inheritance.bases.length === 0" class="empty-section">
              No base classes
            </div>
          </div>
        </InspectorSection>

        <!-- Decorators Section -->
        <InspectorSection
          title="Decorators"
          :is-collapsed="sectionState.decorators"
          :item-count="inspectorData.decorators.length"
          @toggle="toggleSection('decorators')"
        >
          <div class="decorators-list">
            <div
              v-for="(decorator, index) in inspectorData.decorators"
              :key="index"
              class="decorator-item"
            >
              <code
                class="decorator-code"
                :style="{ color: getDecoratorColor(decorator.category) }"
              >
                {{ formatDecorator(decorator) }}
              </code>
            </div>
            <div v-if="inspectorData.decorators.length === 0" class="empty-section">
              No decorators
            </div>
          </div>
        </InspectorSection>

        <!-- Metaclass Section -->
        <InspectorSection
          v-if="inspectorData.metaclass"
          title="Metaclass"
          :is-collapsed="sectionState.metaclass"
          @toggle="toggleSection('metaclass')"
        >
          <div class="metaclass-info">
            <div class="metaclass-name">
              <code :class="{ 'trinity-meta': inspectorData.metaclass.isTrinityMeta }">
                {{ inspectorData.metaclass.name }}
              </code>
            </div>
            <p v-if="inspectorData.metaclass.module" class="metaclass-module">
              from {{ inspectorData.metaclass.module }}
            </p>
            <p v-if="inspectorData.metaclass.description" class="metaclass-desc">
              {{ inspectorData.metaclass.description }}
            </p>
          </div>
        </InspectorSection>

        <!-- Fields Section -->
        <InspectorSection
          title="Fields"
          :is-collapsed="sectionState.fields"
          :item-count="inspectorData.fields.length"
          @toggle="toggleSection('fields')"
        >
          <div class="fields-list">
            <div
              v-for="field in inspectorData.fields"
              :key="field.name"
              class="field-item"
            >
              <div class="field-header">
                <span class="field-name">{{ field.name }}</span>
                <span class="field-type">{{ field.type }}</span>
              </div>
              <div v-if="field.hasDefault" class="field-default">
                <span class="default-label">default:</span>
                <code class="default-value">{{ field.default }}</code>
              </div>
              <p v-if="field.doc" class="field-doc">{{ field.doc }}</p>
            </div>
            <div v-if="inspectorData.fields.length === 0" class="empty-section">
              No fields
            </div>
          </div>
        </InspectorSection>

        <!-- Methods Section -->
        <InspectorSection
          title="Methods"
          :is-collapsed="sectionState.methods"
          :item-count="inspectorData.methods.length"
          @toggle="toggleSection('methods')"
        >
          <div class="methods-list">
            <div
              v-for="method in inspectorData.methods"
              :key="method.name"
              class="method-item"
            >
              <div class="method-badges">
                <span v-if="method.isStatic" class="method-badge static">static</span>
                <span v-if="method.isClassMethod" class="method-badge classmethod">classmethod</span>
                <span v-if="method.isProperty" class="method-badge property">property</span>
                <span v-if="method.isAbstract" class="method-badge abstract">abstract</span>
              </div>
              <code class="method-signature">{{ formatMethodSignature(method) }}</code>
              <p v-if="method.doc" class="method-doc">{{ method.doc }}</p>
            </div>
            <div v-if="inspectorData.methods.length === 0" class="empty-section">
              No methods
            </div>
          </div>
        </InspectorSection>

        <!-- Source Section (Full Docstring) -->
        <InspectorSection
          v-if="inspectorData.docstring && inspectorData.docstring.length > 100"
          title="Documentation"
          :is-collapsed="sectionState.source"
          @toggle="toggleSection('source')"
        >
          <div class="full-docstring">
            <pre class="docstring-content">{{ inspectorData.docstring }}</pre>
          </div>
        </InspectorSection>

        <!-- Quick Info Footer -->
        <div class="quick-info-footer">
          <div class="info-item" :title="'Full qualified name: ' + inspectorData.qualifiedName">
            <span class="info-label">Qualified Name:</span>
            <code class="info-value">{{ inspectorData.qualifiedName }}</code>
          </div>
          <div v-if="inspectorData.inheritance.isMultipleInheritance" class="info-item info-warning">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            <span>Multiple Inheritance</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useInspector } from '@/composables/useInspector'
import InspectorSection from './InspectorSection.vue'

// =============================================================================
// PROPS & EMITS
// =============================================================================

const emit = defineEmits<{
  (e: 'open-in-editor', file: string, line: number): void
}>()

// =============================================================================
// COMPOSABLE
// =============================================================================

const {
  inspectorData,
  isLoading,
  error,
  hasSelection,
  sectionState,
  toggleSection,
  formatDecorator,
  formatMethodSignature,
  getDecoratorColor,
  getTrinityTypeColor,
  openInEditor,
} = useInspector({
  autoFetch: true,
  onDataLoaded: (data) => {
    console.log('[InspectorPanel] Data loaded for:', data.className)
  },
  onError: (err) => {
    console.error('[InspectorPanel] Error:', err.message)
  },
})

// =============================================================================
// METHODS
// =============================================================================

/**
 * Format source path for display (truncate if too long).
 */
function formatSourcePath(path: string): string {
  const maxLen = 30
  if (path.length <= maxLen) {
    return path
  }

  const parts = path.split('/')
  const fileName = parts[parts.length - 1]

  if (fileName && fileName.length < maxLen - 4) {
    return `.../${fileName}`
  }

  return '...' + path.slice(-(maxLen - 3))
}

/**
 * Handle View Source button click.
 */
function handleViewSource(): void {
  if (!inspectorData.value) return

  openInEditor(inspectorData.value.source)
  emit('open-in-editor', inspectorData.value.source.file, inspectorData.value.source.line)
}
</script>

<style scoped>
.inspector-panel {
  display: flex;
  flex-direction: column;
  width: 320px;
  height: 100%;
  background-color: var(--flowforge-panel-bg, #252530);
  border-left: 1px solid var(--flowforge-border, #3a3a4a);
  overflow: hidden;
}

/* Header */
.inspector-header {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
  flex-shrink: 0;
}

.inspector-title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--flowforge-text-bright, #ffffff);
}

/* Empty State */
.inspector-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  padding: 32px 24px;
  text-align: center;
}

.empty-icon {
  color: var(--flowforge-text-muted, #666680);
  margin-bottom: 16px;
  opacity: 0.6;
}

.empty-text {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: var(--flowforge-text-secondary, #a0a0a0);
}

.empty-hint {
  margin: 0;
  font-size: 12px;
  color: var(--flowforge-text-muted, #666680);
}

/* Loading State */
.inspector-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  padding: 32px 24px;
}

.loading-spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--flowforge-border, #3a3a4a);
  border-top-color: var(--flowforge-primary, #6366f1);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-text {
  margin: 12px 0 0 0;
  font-size: 13px;
  color: var(--flowforge-text-secondary, #a0a0a0);
}

/* Error State */
.inspector-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  padding: 32px 24px;
  text-align: center;
}

.error-icon {
  color: var(--flowforge-danger, #dc2626);
  margin-bottom: 12px;
}

.error-text {
  margin: 0;
  font-size: 13px;
  color: var(--flowforge-danger, #dc2626);
}

/* Inspector Content */
.inspector-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow-y: auto;
}

/* Class Header */
.class-header {
  padding: 16px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.class-name-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.trinity-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  color: var(--flowforge-text-bright, #ffffff);
}

.class-name {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--flowforge-text-bright, #ffffff);
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
}

.module-name {
  margin: 0;
  font-size: 12px;
  color: var(--flowforge-text-muted, #666680);
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
}

.docstring {
  margin: 12px 0 0 0;
  padding: 8px;
  background-color: var(--flowforge-surface, #2a2a3a);
  border-radius: 4px;
  font-size: 12px;
  color: var(--flowforge-text-secondary, #a0a0a0);
  font-style: italic;
}

/* Source Location */
.source-location {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
  background-color: var(--flowforge-surface, #2a2a3a);
}

.source-info {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  min-width: 0;
}

.source-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--flowforge-text-muted, #666680);
  flex-shrink: 0;
}

.source-path {
  font-size: 11px;
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  color: var(--flowforge-text-secondary, #a0a0a0);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.source-line {
  font-size: 11px;
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  color: var(--flowforge-primary-light, #818cf8);
  flex-shrink: 0;
}

.open-editor-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background-color: var(--flowforge-primary, #6366f1);
  border: 1px solid var(--flowforge-primary, #6366f1);
  border-radius: 4px;
  color: var(--flowforge-text-bright, #ffffff);
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.open-editor-btn:hover {
  background-color: var(--flowforge-primary-dark, #4f46e5);
  border-color: var(--flowforge-primary-dark, #4f46e5);
}

.open-editor-btn:active {
  transform: scale(0.98);
}

/* Sections */
.inspector-sections {
  flex: 1;
  overflow-y: auto;
}

/* Inheritance */
.inheritance-chain {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.base-class {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  color: var(--flowforge-text, #e0e0e0);
}

.base-class.trinity-base {
  color: var(--flowforge-primary-light, #818cf8);
}

.base-icon {
  display: flex;
  color: var(--flowforge-text-muted, #666680);
}

.base-name {
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12px;
}

.base-module {
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
}

/* Decorators */
.decorators-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.decorator-item {
  padding: 2px 0;
}

.decorator-code {
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12px;
}

/* Metaclass */
.metaclass-info {
  padding: 4px 0;
}

.metaclass-name code {
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12px;
  color: var(--flowforge-text, #e0e0e0);
}

.metaclass-name code.trinity-meta {
  color: var(--flowforge-primary-light, #818cf8);
}

.metaclass-module {
  margin: 4px 0 0 0;
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
}

.metaclass-desc {
  margin: 8px 0 0 0;
  font-size: 12px;
  color: var(--flowforge-text-secondary, #a0a0a0);
  font-style: italic;
}

/* Fields */
.fields-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.field-item {
  padding: 6px 0;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.field-item:last-child {
  border-bottom: none;
}

.field-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.field-name {
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12px;
  font-weight: 500;
  color: var(--flowforge-text, #e0e0e0);
}

.field-type {
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 11px;
  color: var(--flowforge-primary-light, #818cf8);
}

.field-default {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
}

.default-label {
  font-size: 10px;
  color: var(--flowforge-text-muted, #666680);
}

.default-value {
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 11px;
  color: var(--flowforge-text-secondary, #a0a0a0);
}

.field-doc {
  margin: 4px 0 0 0;
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
  font-style: italic;
}

/* Methods */
.methods-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.method-item {
  padding: 6px 0;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.method-item:last-child {
  border-bottom: none;
}

.method-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 4px;
}

.method-badge {
  display: inline-flex;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 9px;
  font-weight: 500;
  text-transform: uppercase;
}

.method-badge.static {
  background-color: rgba(168, 85, 247, 0.2);
  color: var(--flowforge-node-resource, #a855f7);
}

.method-badge.classmethod {
  background-color: rgba(34, 197, 94, 0.2);
  color: var(--flowforge-node-system, #22c55e);
}

.method-badge.property {
  background-color: rgba(59, 130, 246, 0.2);
  color: var(--flowforge-node-component, #3b82f6);
}

.method-badge.abstract {
  background-color: rgba(249, 115, 22, 0.2);
  color: var(--flowforge-node-event, #f97316);
}

.method-signature {
  display: block;
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 11px;
  color: var(--flowforge-text, #e0e0e0);
  word-break: break-all;
}

.method-doc {
  margin: 4px 0 0 0;
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
  font-style: italic;
}

/* Empty Section */
.empty-section {
  padding: 8px 0;
  font-size: 12px;
  color: var(--flowforge-text-muted, #666680);
  font-style: italic;
}

/* Full Docstring Section */
.full-docstring {
  padding: 4px 0;
}

.docstring-content {
  margin: 0;
  padding: 8px;
  background-color: var(--flowforge-surface, #2a2a3a);
  border-radius: 4px;
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 11px;
  color: var(--flowforge-text-secondary, #a0a0a0);
  white-space: pre-wrap;
  word-break: break-word;
  overflow-x: auto;
}

/* Quick Info Footer */
.quick-info-footer {
  padding: 12px 16px;
  border-top: 1px solid var(--flowforge-border, #3a3a4a);
  background-color: var(--flowforge-surface, #2a2a3a);
  flex-shrink: 0;
}

.info-item {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.info-item:last-child {
  margin-bottom: 0;
}

.info-label {
  font-size: 10px;
  font-weight: 500;
  color: var(--flowforge-text-muted, #666680);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.info-value {
  font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 11px;
  color: var(--flowforge-text-secondary, #a0a0a0);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.info-warning {
  color: var(--flowforge-warning, #f59e0b);
  font-size: 11px;
}

.info-warning svg {
  flex-shrink: 0;
}
</style>
