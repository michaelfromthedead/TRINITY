<template>
  <div class="node-palette">
    <!-- Search Box -->
    <div class="palette-search">
      <SearchBox
        v-model="searchQuery"
        placeholder="Search nodes..."
        @update:model-value="handleSearch"
      />
    </div>

    <!-- Loading State -->
    <div v-if="isLoading" class="palette-loading">
      <span>Loading node types...</span>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="palette-error">
      <span>{{ error }}</span>
      <button class="retry-btn" @click="loadNodeTypes">Retry</button>
    </div>

    <!-- Search Results -->
    <div v-else-if="searchQuery && searchResults.length > 0" class="palette-section">
      <div class="section-header">
        <span class="section-title">Search Results</span>
        <span class="section-count">{{ searchResults.length }}</span>
      </div>
      <div class="node-list">
        <NodePaletteItem
          v-for="node in searchResults"
          :key="node.type"
          :node="node"
          @dragstart="handleDragStart($event, node)"
          @click="handleNodeClick(node)"
        />
      </div>
    </div>

    <!-- No Results -->
    <div v-else-if="searchQuery && searchResults.length === 0" class="palette-empty">
      <span>No nodes found for "{{ searchQuery }}"</span>
    </div>

    <!-- Categories -->
    <template v-else>
      <!-- Trinity ECS Section -->
      <div class="palette-section trinity-section">
        <div class="section-header" @click="toggleSection('trinity')">
          <i class="pi pi-box section-icon" />
          <span class="section-title">Trinity ECS</span>
          <i :class="['pi', expandedSections.trinity ? 'pi-chevron-down' : 'pi-chevron-right', 'section-chevron']" />
        </div>
        <div v-show="expandedSections.trinity" class="section-content">
          <div class="subsection">
            <div class="subsection-header">
              <span class="subsection-dot component-dot" />
              <span>Components</span>
            </div>
            <div class="node-list">
              <NodePaletteItem
                v-for="node in componentNodes"
                :key="node.type"
                :node="node"
                @dragstart="handleDragStart($event, node)"
                @click="handleNodeClick(node)"
              />
            </div>
          </div>
          <div class="subsection">
            <div class="subsection-header">
              <span class="subsection-dot system-dot" />
              <span>Systems</span>
            </div>
            <div class="node-list">
              <NodePaletteItem
                v-for="node in systemNodes"
                :key="node.type"
                :node="node"
                @dragstart="handleDragStart($event, node)"
                @click="handleNodeClick(node)"
              />
            </div>
          </div>
          <div class="subsection">
            <div class="subsection-header">
              <span class="subsection-dot resource-dot" />
              <span>Resources</span>
            </div>
            <div class="node-list">
              <NodePaletteItem
                v-for="node in resourceNodes"
                :key="node.type"
                :node="node"
                @dragstart="handleDragStart($event, node)"
                @click="handleNodeClick(node)"
              />
            </div>
          </div>
          <div class="subsection">
            <div class="subsection-header">
              <span class="subsection-dot event-dot" />
              <span>Events</span>
            </div>
            <div class="node-list">
              <NodePaletteItem
                v-for="node in eventNodes"
                :key="node.type"
                :node="node"
                @dragstart="handleDragStart($event, node)"
                @click="handleNodeClick(node)"
              />
            </div>
          </div>
        </div>
      </div>

      <!-- Other Categories -->
      <div
        v-for="category in otherCategories"
        :key="category.id"
        class="palette-section"
      >
        <div class="section-header" @click="toggleSection(category.id)">
          <i :class="[category.icon || 'pi pi-folder', 'section-icon']" />
          <span class="section-title">{{ category.name }}</span>
          <span class="section-count">{{ category.nodeTypes.length }}</span>
          <i :class="['pi', expandedSections[category.id] ? 'pi-chevron-down' : 'pi-chevron-right', 'section-chevron']" />
        </div>
        <div v-show="expandedSections[category.id]" class="section-content">
          <div class="node-list">
            <NodePaletteItem
              v-for="nodeType in category.nodeTypes"
              :key="nodeType"
              :node="getNodeDef(nodeType)"
              @dragstart="handleDragStart($event, getNodeDef(nodeType))"
              @click="handleNodeClick(getNodeDef(nodeType))"
            />
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, reactive } from 'vue'
import SearchBox from '@/components/common/SearchBox.vue'
import NodePaletteItem from './NodePaletteItem.vue'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import type { NodeTypeDefinition } from '@/services'

// =============================================================================
// EMITS
// =============================================================================

const emit = defineEmits<{
  (e: 'nodeSelected', nodeType: string): void
  (e: 'nodeDragStart', nodeType: string, event: DragEvent): void
}>()

// =============================================================================
// STATE
// =============================================================================

const nodeDefStore = useNodeDefStore()
const searchQuery = ref('')
const searchResults = ref<any[]>([])

const expandedSections = reactive<Record<string, boolean>>({
  trinity: true,
  math: false,
  logic: false
})

// =============================================================================
// COMPUTED
// =============================================================================

const isLoading = computed(() => nodeDefStore.loading)
const error = computed(() => nodeDefStore.error)

// Trinity node categories
const componentNodes = computed(() => {
  const def = nodeDefStore.componentDef
  return def ? [{ type: 'trinity/component', ...def }] : []
})

const systemNodes = computed(() => {
  const def = nodeDefStore.systemDef
  return def ? [{ type: 'trinity/system', ...def }] : []
})

const resourceNodes = computed(() => {
  const def = nodeDefStore.resourceDef
  return def ? [{ type: 'trinity/resource', ...def }] : []
})

const eventNodes = computed(() => {
  const def = nodeDefStore.eventDef
  return def ? [{ type: 'trinity/event', ...def }] : []
})

// Other categories (non-Trinity)
const otherCategories = computed(() => {
  // This would come from a more complete node definition store
  return [
    {
      id: 'math',
      name: 'Math',
      icon: 'pi pi-calculator',
      nodeTypes: ['Math/Add', 'Math/Subtract', 'Math/Multiply', 'Math/Divide']
    },
    {
      id: 'logic',
      name: 'Logic',
      icon: 'pi pi-sitemap',
      nodeTypes: ['Logic/Branch', 'Logic/Compare', 'Logic/Gate']
    }
  ]
})

// =============================================================================
// METHODS
// =============================================================================

function getNodeDef(nodeType: string): any {
  // For Trinity nodes
  const parts = nodeType.split('/')
  if (parts[0] === 'trinity') {
    const typeName = parts[1]?.toLowerCase()
    const def = nodeDefStore.getNodeDef(typeName)
    return def ? { type: nodeType, ...def } : { type: nodeType, name: parts[1] || nodeType }
  }

  // For other nodes, return a placeholder
  return {
    type: nodeType,
    name: parts[parts.length - 1] || nodeType,
    description: `${nodeType} node`
  }
}

function toggleSection(sectionId: string) {
  expandedSections[sectionId] = !expandedSections[sectionId]
}

function handleSearch(query: string) {
  if (!query.trim()) {
    searchResults.value = []
    return
  }

  // Search across all available nodes
  const results: any[] = []
  const lowerQuery = query.toLowerCase()

  // Search Trinity nodes
  const trinityTypes = ['component', 'system', 'resource', 'event']
  for (const typeName of trinityTypes) {
    const def = nodeDefStore.getNodeDef(typeName)
    if (def) {
      const nodeType = `trinity/${typeName}`
      const matches =
        typeName.includes(lowerQuery) ||
        def.name?.toLowerCase().includes(lowerQuery) ||
        def.description?.toLowerCase().includes(lowerQuery)

      if (matches) {
        results.push({ type: nodeType, ...def })
      }
    }
  }

  // Search other categories
  for (const category of otherCategories.value) {
    for (const nodeType of category.nodeTypes) {
      const nodeDef = getNodeDef(nodeType)
      const matches =
        nodeType.toLowerCase().includes(lowerQuery) ||
        nodeDef.name?.toLowerCase().includes(lowerQuery)

      if (matches) {
        results.push(nodeDef)
      }
    }
  }

  searchResults.value = results
}

function handleDragStart(event: DragEvent, node: any) {
  if (!event.dataTransfer || !node) return

  event.dataTransfer.setData('application/flowforge-node', JSON.stringify({
    type: node.type,
    name: node.name || node.type
  }))
  event.dataTransfer.effectAllowed = 'copy'

  emit('nodeDragStart', node.type, event)
}

function handleNodeClick(node: any) {
  if (!node) return
  emit('nodeSelected', node.type)
}

async function loadNodeTypes() {
  await nodeDefStore.loadNodeTypes()
}

// =============================================================================
// LIFECYCLE
// =============================================================================

onMounted(async () => {
  if (!nodeDefStore.isLoaded) {
    await loadNodeTypes()
  }
})
</script>

<style scoped>
.node-palette {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: var(--panel-bg, #252526);
  color: var(--text-primary, #cccccc);
  overflow: hidden;
}

.palette-search {
  padding: 8px;
  border-bottom: 1px solid var(--border-color, #3d3d3d);
}

.palette-loading,
.palette-error,
.palette-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px 16px;
  color: var(--text-muted, #888);
  font-size: 13px;
  text-align: center;
  gap: 12px;
}

.palette-error {
  color: var(--error-color, #f44336);
}

.retry-btn {
  padding: 6px 12px;
  border: 1px solid var(--border-color, #3d3d3d);
  border-radius: 4px;
  background: transparent;
  color: var(--text-primary, #cccccc);
  cursor: pointer;
  font-size: 12px;
}

.retry-btn:hover {
  background-color: var(--hover-bg, #2a2d2e);
}

.palette-section {
  border-bottom: 1px solid var(--border-color, #3d3d3d);
}

.section-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  cursor: pointer;
  user-select: none;
}

.section-header:hover {
  background-color: var(--hover-bg, #2a2d2e);
}

.section-icon {
  font-size: 14px;
  color: var(--text-muted, #888);
}

.section-title {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
}

.section-count {
  font-size: 11px;
  color: var(--text-muted, #888);
  background-color: var(--badge-bg, #333);
  padding: 2px 6px;
  border-radius: 10px;
}

.section-chevron {
  font-size: 10px;
  color: var(--text-muted, #888);
}

.section-content {
  padding: 4px 0;
}

.subsection {
  padding: 4px 0;
}

.subsection-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  font-size: 12px;
  color: var(--text-muted, #888);
}

.subsection-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.component-dot {
  background-color: var(--trinity-component, #4CAF50);
}

.system-dot {
  background-color: var(--trinity-system, #2196F3);
}

.resource-dot {
  background-color: var(--trinity-resource, #FF9800);
}

.event-dot {
  background-color: var(--trinity-event, #9C27B0);
}

.node-list {
  display: flex;
  flex-direction: column;
}

/* Trinity section special styling */
.trinity-section .section-header {
  background-color: var(--trinity-section-bg, rgba(33, 150, 243, 0.1));
}

.trinity-section .section-icon {
  color: var(--trinity-accent, #2196F3);
}
</style>
