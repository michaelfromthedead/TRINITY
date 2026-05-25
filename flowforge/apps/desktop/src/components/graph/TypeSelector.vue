<template>
  <Teleport to="body">
    <Transition name="fade-slide">
      <div
        v-if="visible"
        ref="selectorRef"
        class="type-selector-overlay"
        :style="overlayStyle"
        @click.stop
        @mousedown.stop
        @pointerdown.stop
      >
        <div class="type-selector-container">
          <!-- Search Input -->
          <div class="type-selector-search">
            <input
              ref="searchInputRef"
              v-model="searchQuery"
              type="text"
              class="type-selector-search-input"
              placeholder="Search or enter type..."
              @keydown.enter.prevent="handleSelectCurrent"
              @keydown.escape.prevent="handleCancel"
              @keydown.down.prevent="navigateDown"
              @keydown.up.prevent="navigateUp"
              @input="handleSearchInput"
            />
          </div>

          <!-- Type List -->
          <div class="type-selector-list" @wheel.stop>
            <!-- Python Built-in Types -->
            <div v-if="filteredBuiltinTypes.length > 0" class="type-selector-section">
              <div class="type-selector-section-header">Python Types</div>
              <button
                v-for="(type, index) in filteredBuiltinTypes"
                :key="`builtin-${type}`"
                class="type-selector-item"
                :class="{ selected: isSelected('builtin', index) }"
                @click="handleSelect(type)"
                @mouseenter="setHoverIndex('builtin', index)"
              >
                <span class="type-icon builtin">T</span>
                <span class="type-name">{{ type }}</span>
              </button>
            </div>

            <!-- Typing Module Types -->
            <div v-if="filteredTypingTypes.length > 0" class="type-selector-section">
              <div class="type-selector-section-header">Generic Types</div>
              <button
                v-for="(type, index) in filteredTypingTypes"
                :key="`typing-${type}`"
                class="type-selector-item"
                :class="{ selected: isSelected('typing', index) }"
                @click="handleSelect(type)"
                @mouseenter="setHoverIndex('typing', index)"
              >
                <span class="type-icon typing">G</span>
                <span class="type-name">{{ type }}</span>
                <span v-if="isGenericType(type)" class="type-hint">[T]</span>
              </button>
            </div>

            <!-- Trinity Types -->
            <div v-if="filteredTrinityTypes.length > 0" class="type-selector-section">
              <div class="type-selector-section-header">Trinity Types</div>
              <button
                v-for="(type, index) in filteredTrinityTypes"
                :key="`trinity-${type}`"
                class="type-selector-item"
                :class="{ selected: isSelected('trinity', index) }"
                @click="handleSelect(type)"
                @mouseenter="setHoverIndex('trinity', index)"
              >
                <span class="type-icon trinity">E</span>
                <span class="type-name">{{ type }}</span>
                <span v-if="type.includes('[')">
                  <span class="type-hint">[T]</span>
                </span>
              </button>
            </div>

            <!-- Custom Type Option -->
            <div v-if="showCustomOption" class="type-selector-section">
              <div class="type-selector-section-header">Custom</div>
              <button
                class="type-selector-item custom"
                :class="{ selected: isCustomSelected }"
                @click="handleSelectCustom"
                @mouseenter="setCustomHover"
              >
                <span class="type-icon custom">+</span>
                <span class="type-name">{{ searchQuery }}</span>
                <span class="type-hint">(custom)</span>
              </button>
            </div>

            <!-- No Results -->
            <div v-if="noResults" class="type-selector-empty">
              <span>No matching types</span>
              <span v-if="searchQuery" class="type-selector-empty-hint">
                Press Enter to use "{{ searchQuery }}"
              </span>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch, type CSSProperties } from 'vue'
import {
  PYTHON_BUILTIN_TYPES,
  TYPING_TYPES,
  TRINITY_DATA_TYPES
} from '@/composables/useInlineEdit'

// =============================================================================
// Props & Emits
// =============================================================================

export interface TypeSelectorProps {
  /** Whether the selector is visible */
  visible?: boolean
  /** Currently selected type */
  value?: string
  /** Position for the dropdown */
  position?: {
    x: number
    y: number
    width: number
  } | null
  /** Maximum height of the dropdown */
  maxHeight?: number
}

const props = withDefaults(defineProps<TypeSelectorProps>(), {
  visible: false,
  value: '',
  position: null,
  maxHeight: 300
})

const emit = defineEmits<{
  (e: 'update:value', value: string): void
  (e: 'select', value: string): void
  (e: 'cancel'): void
}>()

// =============================================================================
// Refs
// =============================================================================

const selectorRef = ref<HTMLElement | null>(null)
const searchInputRef = ref<HTMLInputElement | null>(null)
const searchQuery = ref('')
const selectedSection = ref<'builtin' | 'typing' | 'trinity' | 'custom' | null>(null)
const selectedIndex = ref(0)

// =============================================================================
// Computed
// =============================================================================

const visible = computed(() => props.visible)

const overlayStyle = computed<CSSProperties>(() => {
  if (!props.position) {
    return { display: 'none' }
  }

  return {
    position: 'fixed',
    left: `${props.position.x}px`,
    top: `${props.position.y}px`,
    width: `${Math.max(props.position.width, 200)}px`,
    maxHeight: `${props.maxHeight}px`,
    zIndex: 10000
  }
})

const filteredBuiltinTypes = computed(() => {
  const query = searchQuery.value.toLowerCase()
  if (!query) return [...PYTHON_BUILTIN_TYPES]
  return PYTHON_BUILTIN_TYPES.filter(t => t.toLowerCase().includes(query))
})

const filteredTypingTypes = computed(() => {
  const query = searchQuery.value.toLowerCase()
  if (!query) return [...TYPING_TYPES]
  return TYPING_TYPES.filter(t => t.toLowerCase().includes(query))
})

const filteredTrinityTypes = computed(() => {
  const query = searchQuery.value.toLowerCase()
  if (!query) return [...TRINITY_DATA_TYPES]
  return TRINITY_DATA_TYPES.filter(t => t.toLowerCase().includes(query))
})

const allFilteredTypes = computed(() => [
  ...filteredBuiltinTypes.value.map(t => ({ type: t, section: 'builtin' as const })),
  ...filteredTypingTypes.value.map(t => ({ type: t, section: 'typing' as const })),
  ...filteredTrinityTypes.value.map(t => ({ type: t, section: 'trinity' as const }))
])

const noResults = computed(() => allFilteredTypes.value.length === 0)

const showCustomOption = computed(() => {
  const query = searchQuery.value.trim()
  if (!query) return false
  // Show custom option if query doesn't exactly match any known type
  const allTypes = [
    ...PYTHON_BUILTIN_TYPES,
    ...TYPING_TYPES,
    ...TRINITY_DATA_TYPES
  ]
  return !allTypes.includes(query as typeof PYTHON_BUILTIN_TYPES[number])
})

const isCustomSelected = computed(() => {
  return selectedSection.value === 'custom'
})

// =============================================================================
// Methods
// =============================================================================

function isSelected(section: 'builtin' | 'typing' | 'trinity', index: number): boolean {
  return selectedSection.value === section && selectedIndex.value === index
}

function isGenericType(type: string): boolean {
  return ['Optional', 'Union', 'List', 'Dict', 'Set', 'Tuple', 'Callable', 'Query', 'Resource'].includes(type)
}

function setHoverIndex(section: 'builtin' | 'typing' | 'trinity', index: number): void {
  selectedSection.value = section
  selectedIndex.value = index
}

function setCustomHover(): void {
  selectedSection.value = 'custom'
  selectedIndex.value = 0
}

function handleSearchInput(): void {
  // Reset selection when search changes
  if (allFilteredTypes.value.length > 0) {
    selectedSection.value = allFilteredTypes.value[0].section
    selectedIndex.value = 0
  } else if (showCustomOption.value) {
    selectedSection.value = 'custom'
    selectedIndex.value = 0
  } else {
    selectedSection.value = null
    selectedIndex.value = 0
  }
}

function navigateDown(): void {
  const types = allFilteredTypes.value
  if (types.length === 0) {
    if (showCustomOption.value) {
      selectedSection.value = 'custom'
    }
    return
  }

  // Find current position in flat list
  let currentPos = types.findIndex(
    (t, i) => t.section === selectedSection.value && getIndexInSection(t.section, i) === selectedIndex.value
  )

  if (currentPos === -1) {
    currentPos = -1
  }

  const nextPos = currentPos + 1
  if (nextPos < types.length) {
    const next = types[nextPos]
    selectedSection.value = next.section
    selectedIndex.value = getIndexInSection(next.section, nextPos)
  } else if (showCustomOption.value) {
    selectedSection.value = 'custom'
    selectedIndex.value = 0
  }
}

function navigateUp(): void {
  const types = allFilteredTypes.value
  if (types.length === 0) return

  if (selectedSection.value === 'custom') {
    if (types.length > 0) {
      const last = types[types.length - 1]
      selectedSection.value = last.section
      selectedIndex.value = getIndexInSection(last.section, types.length - 1)
    }
    return
  }

  // Find current position in flat list
  let currentPos = types.findIndex(
    (t, i) => t.section === selectedSection.value && getIndexInSection(t.section, i) === selectedIndex.value
  )

  if (currentPos === -1) {
    currentPos = types.length
  }

  const prevPos = currentPos - 1
  if (prevPos >= 0) {
    const prev = types[prevPos]
    selectedSection.value = prev.section
    selectedIndex.value = getIndexInSection(prev.section, prevPos)
  }
}

function getIndexInSection(section: 'builtin' | 'typing' | 'trinity', flatIndex: number): number {
  const types = allFilteredTypes.value
  let sectionStart = 0
  for (let i = 0; i < flatIndex; i++) {
    if (types[i].section === section) {
      sectionStart = i
      break
    }
  }
  // Count how many of this section came before
  let count = 0
  for (let i = 0; i <= flatIndex; i++) {
    if (types[i].section === section) {
      if (i === flatIndex) return count
      count++
    }
  }
  return 0
}

function handleSelect(type: string): void {
  emit('select', type)
  emit('update:value', type)
}

function handleSelectCustom(): void {
  const customType = searchQuery.value.trim()
  if (customType) {
    emit('select', customType)
    emit('update:value', customType)
  }
}

function handleSelectCurrent(): void {
  if (selectedSection.value === 'custom') {
    handleSelectCustom()
    return
  }

  const types = allFilteredTypes.value
  if (types.length === 0) {
    // No matches - use search query as custom type
    handleSelectCustom()
    return
  }

  // Find selected type
  let selected: string | null = null
  switch (selectedSection.value) {
    case 'builtin':
      selected = filteredBuiltinTypes.value[selectedIndex.value]
      break
    case 'typing':
      selected = filteredTypingTypes.value[selectedIndex.value]
      break
    case 'trinity':
      selected = filteredTrinityTypes.value[selectedIndex.value]
      break
  }

  if (selected) {
    handleSelect(selected)
  } else if (searchQuery.value.trim()) {
    handleSelectCustom()
  }
}

function handleCancel(): void {
  emit('cancel')
}

function focusInput(): void {
  searchInputRef.value?.focus()
}

// =============================================================================
// Watchers
// =============================================================================

watch(
  () => props.visible,
  async (isVisible) => {
    if (isVisible) {
      searchQuery.value = props.value || ''
      selectedSection.value = null
      selectedIndex.value = 0
      await nextTick()
      focusInput()
      handleSearchInput()
    }
  }
)

watch(
  () => props.value,
  (newValue) => {
    if (props.visible && newValue) {
      searchQuery.value = newValue
    }
  }
)

// =============================================================================
// Expose
// =============================================================================

defineExpose({
  focus: focusInput,
  searchInputRef
})
</script>

<style scoped>
.type-selector-overlay {
  pointer-events: auto;
}

.type-selector-container {
  background: var(--p-surface-0, #1e1e1e);
  border: 1px solid var(--p-surface-300, #404040);
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.type-selector-search {
  padding: 8px;
  border-bottom: 1px solid var(--p-surface-200, #333);
}

.type-selector-search-input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--p-surface-300, #404040);
  border-radius: 4px;
  background: var(--p-surface-50, #2a2a2a);
  color: var(--p-text-color, #e5e5e5);
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12px;
  outline: none;
  transition: border-color 0.15s ease;
}

.type-selector-search-input:focus {
  border-color: var(--p-primary-500, #3b82f6);
}

.type-selector-search-input::placeholder {
  color: var(--p-text-muted-color, #6b7280);
}

.type-selector-list {
  max-height: 250px;
  overflow-y: auto;
  padding: 4px;
}

.type-selector-section {
  margin-bottom: 4px;
}

.type-selector-section:last-child {
  margin-bottom: 0;
}

.type-selector-section-header {
  padding: 4px 8px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  color: var(--p-text-muted-color, #6b7280);
  letter-spacing: 0.5px;
}

.type-selector-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 6px 8px;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--p-text-color, #e5e5e5);
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.1s ease;
}

.type-selector-item:hover,
.type-selector-item.selected {
  background: var(--p-surface-100, #333);
}

.type-selector-item.selected {
  background: var(--p-primary-500, #3b82f6);
  color: white;
}

.type-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: bold;
}

.type-icon.builtin {
  background: var(--p-blue-500, #3b82f6);
  color: white;
}

.type-icon.typing {
  background: var(--p-purple-500, #a855f7);
  color: white;
}

.type-icon.trinity {
  background: var(--p-green-500, #22c55e);
  color: white;
}

.type-icon.custom {
  background: var(--p-orange-500, #f97316);
  color: white;
}

.type-name {
  flex: 1;
}

.type-hint {
  font-size: 10px;
  color: var(--p-text-muted-color, #6b7280);
}

.type-selector-item.selected .type-hint {
  color: rgba(255, 255, 255, 0.7);
}

.type-selector-item.custom {
  border: 1px dashed var(--p-surface-300, #404040);
}

.type-selector-empty {
  padding: 16px;
  text-align: center;
  color: var(--p-text-muted-color, #6b7280);
  font-size: 12px;
}

.type-selector-empty-hint {
  display: block;
  margin-top: 4px;
  font-size: 11px;
  color: var(--p-text-muted-color, #6b7280);
}

/* Animations */
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: opacity 0.15s ease, transform 0.15s ease;
}

.fade-slide-enter-from,
.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

/* Scrollbar styling */
.type-selector-list::-webkit-scrollbar {
  width: 6px;
}

.type-selector-list::-webkit-scrollbar-track {
  background: transparent;
}

.type-selector-list::-webkit-scrollbar-thumb {
  background: var(--p-surface-300, #404040);
  border-radius: 3px;
}

.type-selector-list::-webkit-scrollbar-thumb:hover {
  background: var(--p-surface-400, #555);
}
</style>
