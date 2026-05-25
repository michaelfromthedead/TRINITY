/**
 * Dialog Store - Generic dialog management for FlowForge
 * Extracted from ComfyUI frontend, stripped of SD-specific code
 */
import { merge } from 'lodash-es'
import { defineStore } from 'pinia'
import { markRaw, ref } from 'vue'
import type { Component } from 'vue'
import { UI_CONFIG } from '@/config/flowforge.config'

type DialogPosition =
  | 'center'
  | 'top'
  | 'bottom'
  | 'left'
  | 'right'
  | 'topleft'
  | 'topright'
  | 'bottomleft'
  | 'bottomright'

export interface DialogComponentProps {
  maximizable?: boolean
  maximized?: boolean
  onClose?: () => void
  closable?: boolean
  modal?: boolean
  position?: DialogPosition
  closeOnEscape?: boolean
  dismissableMask?: boolean
  unstyled?: boolean
  headless?: boolean
}

interface DialogInstance<
  H extends Component = Component,
  B extends Component = Component,
  F extends Component = Component
> {
  key: string
  visible: boolean
  title?: string
  headerComponent?: H
  headerProps?: Record<string, unknown>
  component: B
  contentProps: Record<string, unknown>
  footerComponent?: F
  footerProps?: Record<string, unknown>
  dialogComponentProps: DialogComponentProps
  priority: number
}

export interface ShowDialogOptions<
  H extends Component = Component,
  B extends Component = Component,
  F extends Component = Component
> {
  key?: string
  title?: string
  headerComponent?: H
  footerComponent?: F
  component: B
  props?: Record<string, unknown>
  headerProps?: Record<string, unknown>
  footerProps?: Record<string, unknown>
  dialogComponentProps?: DialogComponentProps
  /**
   * Optional priority for dialog stacking.
   * A dialog will never be shown above a dialog with a higher priority.
   * @default 1
   */
  priority?: number
}

export const useDialogStore = defineStore('dialog', () => {
  const dialogStack = ref<DialogInstance[]>([])

  /**
   * The key of the currently active (top-most) dialog.
   * Only the active dialog can be closed with the ESC key.
   */
  const activeKey = ref<string | null>(null)

  const genDialogKey = () => `dialog-${Math.random().toString(36).slice(2, 9)}`

  /**
   * Inserts a dialog into the stack at the correct position based on priority.
   * Higher priority dialogs are placed before lower priority ones.
   */
  function insertDialogByPriority(dialog: DialogInstance) {
    const insertIndex = dialogStack.value.findIndex(
      (d) => d.priority <= dialog.priority
    )

    dialogStack.value.splice(
      insertIndex === -1 ? dialogStack.value.length : insertIndex,
      0,
      dialog
    )
  }

  function riseDialog(options: { key: string }) {
    const dialogKey = options.key

    const index = dialogStack.value.findIndex((d) => d.key === dialogKey)
    if (index !== -1) {
      const [dialog] = dialogStack.value.splice(index, 1)
      insertDialogByPriority(dialog)
      activeKey.value = dialogKey
      updateCloseOnEscapeStates()
    }
  }

  function closeDialog(options?: { key: string }) {
    const targetDialog = options
      ? dialogStack.value.find((d) => d.key === options.key)
      : dialogStack.value.find((d) => d.key === activeKey.value)
    if (!targetDialog) return

    // Remove from stack first to prevent recursive calls
    // (onClose callback in createDialog also calls closeDialog)
    const index = dialogStack.value.indexOf(targetDialog)
    dialogStack.value.splice(index, 1)

    // Set visible to false so PrimeVue Dialog unmounts cleanly
    targetDialog.visible = false

    activeKey.value =
      dialogStack.value.length > 0
        ? dialogStack.value[dialogStack.value.length - 1].key
        : null

    updateCloseOnEscapeStates()
  }

  function createDialog<
    H extends Component = Component,
    B extends Component = Component,
    F extends Component = Component
  >(options: ShowDialogOptions<H, B, F> & { key: string }) {
    if (dialogStack.value.length >= UI_CONFIG.dialog.maxStack) {
      dialogStack.value.shift()
    }

    const dialog: DialogInstance<H, B, F> = {
      key: options.key,
      visible: true,
      title: options.title,
      headerComponent: options.headerComponent
        ? markRaw(options.headerComponent)
        : undefined,
      footerComponent: options.footerComponent
        ? markRaw(options.footerComponent)
        : undefined,
      component: markRaw(options.component),
      headerProps: { ...options.headerProps },
      contentProps: { ...options.props },
      footerProps: { ...options.footerProps },
      priority: options.priority ?? 1,
      dialogComponentProps: {
        maximizable: false,
        modal: true,
        closable: true,
        closeOnEscape: true,
        dismissableMask: true,
        ...options.dialogComponentProps,
        maximized: false,
        onClose: () => {
          closeDialog({ key: options.key })
        }
      }
    }

    insertDialogByPriority(dialog)
    activeKey.value = options.key
    updateCloseOnEscapeStates()

    return dialog
  }

  /**
   * Ensures only the top-most dialog in the stack can be closed with the Escape key.
   */
  function updateCloseOnEscapeStates() {
    const topDialog = dialogStack.value.find((d) => d.key === activeKey.value)
    const topClosable = topDialog?.dialogComponentProps.closable

    dialogStack.value.forEach((dialog) => {
      dialog.dialogComponentProps = {
        ...dialog.dialogComponentProps,
        closeOnEscape: dialog === topDialog && !!topClosable
      }
    })
  }

  function showDialog<
    H extends Component = Component,
    B extends Component = Component,
    F extends Component = Component
  >(options: ShowDialogOptions<H, B, F>) {
    const dialogKey = options.key || genDialogKey()

    let dialog = dialogStack.value.find((d) => d.key === dialogKey)

    if (dialog) {
      dialog.visible = true
      riseDialog(dialog)
    } else {
      dialog = createDialog({ ...options, key: dialogKey })
    }
    return dialog
  }

  /**
   * Shows a dialog from a third party extension.
   * Explicitly keys extension dialogs with `extension-` prefix.
   */
  function showExtensionDialog(options: ShowDialogOptions & { key: string }) {
    const { key } = options
    if (!key) {
      console.error('Extension dialog key is required')
      return
    }

    const extKey = key.startsWith('extension-') ? key : `extension-${key}`

    const dialog = dialogStack.value.find((d) => d.key === extKey)
    if (!dialog) return createDialog({ ...options, key: extKey })

    dialog.visible = true
    riseDialog(dialog)
    return dialog
  }

  function isDialogOpen(key: string) {
    return dialogStack.value.some((d) => d.key === key)
  }

  return {
    dialogStack,
    riseDialog,
    showDialog,
    closeDialog,
    showExtensionDialog,
    isDialogOpen,
    activeKey
  }
})
