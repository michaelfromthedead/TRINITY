<template>
  <div
    v-if="showInput"
    class="group-title-editor node-title-editor"
    :style="inputStyle"
  >
    <EditableText
      :is-editing="showInput"
      :model-value="editedTitle"
      @edit="onEdit"
    />
  </div>
</template>

<script setup lang="ts">
import { useEventListener } from '@vueuse/core'
import { computed, ref, watch } from 'vue'
import type { CSSProperties } from 'vue'

import EditableText from '@/components/common/EditableText.vue'

// Types for LiteGraph integration
interface LGraphGroup {
  pos: [number, number]
  size: [number, number]
  title: string
  titleHeight: number
  font_size: number
}

interface LGraphNode {
  pos: [number, number]
  width: number
  title: string
  getBounding(): [number, number, number, number]
}

interface CanvasEvent {
  detail: {
    subType: string
    group?: LGraphGroup
    node?: LGraphNode
    originalEvent: {
      canvasY: number
    }
  }
}

// Props for external canvas integration
const props = defineProps<{
  canvas?: {
    ds: { scale: number }
    allow_dragcanvas: boolean
    setDirty: (dirty: boolean, background: boolean) => void
  }
  titleEditorTarget?: LGraphNode | LGraphGroup | null
  nodeTitleHeight?: number
  doubleClickTitleToEdit?: boolean
  groupDoubleClickTitleToEdit?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:titleEditorTarget', target: LGraphNode | LGraphGroup | null): void
}>()

const showInput = ref(false)
const editedTitle = ref('')
const inputPositionStyle = ref<CSSProperties>({})
const inputFontStyle = ref<CSSProperties>({})
const inputStyle = computed<CSSProperties>(() => ({
  ...inputPositionStyle.value,
  ...inputFontStyle.value
}))

const previousCanvasDraggable = ref(true)

const updatePosition = (config: { pos: [number, number]; size: [number, number] }) => {
  const canvas = props.canvas
  if (!canvas) return

  const scale = canvas.ds.scale
  inputPositionStyle.value = {
    position: 'absolute',
    left: `${config.pos[0] * scale}px`,
    top: `${config.pos[1] * scale}px`,
    width: `${config.size[0] * scale}px`,
    height: `${config.size[1] * scale}px`
  }
}

const onEdit = (newValue: string) => {
  if (props.titleEditorTarget && newValue?.trim()) {
    const trimmedTitle = newValue.trim()
    props.titleEditorTarget.title = trimmedTitle
    props.canvas?.setDirty(true, true)
  }
  showInput.value = false
  emit('update:titleEditorTarget', null)
  if (props.canvas) {
    props.canvas.allow_dragcanvas = previousCanvasDraggable.value
  }
}

watch(
  () => props.titleEditorTarget,
  (target) => {
    if (target === null) {
      return
    }
    editedTitle.value = target.title
    showInput.value = true
    const canvas = props.canvas
    if (!canvas) return

    previousCanvasDraggable.value = canvas.allow_dragcanvas
    canvas.allow_dragcanvas = false
    const scale = canvas.ds.scale

    // Check if it's a group (has titleHeight) or a node
    if ('titleHeight' in target && target.titleHeight !== undefined) {
      const group = target as LGraphGroup
      updatePosition({
        pos: group.pos,
        size: [group.size[0], group.titleHeight]
      })
      inputFontStyle.value = { fontSize: `${group.font_size * scale}px` }
    } else {
      const node = target as LGraphNode
      const [x, y] = node.getBounding()
      const nodeTitleHeight = props.nodeTitleHeight ?? 30
      updatePosition({
        pos: [x, y],
        size: [node.width, nodeTitleHeight]
      })
      inputFontStyle.value = { fontSize: `${12 * scale}px` }
    }
  }
)

const canvasEventHandler = (event: Event) => {
  const canvasEvent = event as unknown as CanvasEvent
  if (canvasEvent.detail.subType === 'group-double-click') {
    if (!props.groupDoubleClickTitleToEdit) {
      return
    }

    const group = canvasEvent.detail.group
    if (!group) return
    const [, y] = group.pos

    const e = canvasEvent.detail.originalEvent
    const relativeY = e.canvasY - y
    // Only allow editing if the click is on the title bar
    if (relativeY <= group.titleHeight) {
      emit('update:titleEditorTarget', group)
    }
  } else if (canvasEvent.detail.subType === 'node-double-click') {
    if (!props.doubleClickTitleToEdit) {
      return
    }

    const node = canvasEvent.detail.node
    if (!node) return
    const [, y] = node.pos

    const e = canvasEvent.detail.originalEvent
    const relativeY = e.canvasY - y
    // Only allow editing if the click is on the title bar
    if (relativeY <= 0) {
      emit('update:titleEditorTarget', node)
    }
  }
}

useEventListener(document, 'litegraph:canvas', canvasEventHandler)
</script>

<style scoped>
.group-title-editor.node-title-editor {
  z-index: 9999;
  padding: 0.25rem;
}

:deep(.editable-text) {
  width: 100%;
  height: 100%;
}

:deep(.editable-text input) {
  width: 100%;
  height: 100%;
  /* Override the default font size */
  font-size: inherit;
}
</style>
