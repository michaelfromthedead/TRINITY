/**
 * File Explorer Store - File system navigation for FlowForge
 * Manages workspace browsing, folder expansion, and file selection
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { useGraphStore } from './graphStore'
import type { FileEntry } from '@/bridge/files'
import { listDirectory, getWorkspaceRoot } from '@/bridge/files'

export const useFileExplorerStore = defineStore('fileExplorer', () => {
  // ==========================================================================
  // State
  // ==========================================================================

  /** Root directory path of the workspace */
  const workspaceRoot = ref<string | null>(null)

  /** Current browsing path */
  const currentPath = ref<string | null>(null)

  /** Files and folders in the current path */
  const contents = ref<FileEntry[]>([])

  /** Expanded folder paths for tree view */
  const expandedFolders = ref<Set<string>>(new Set())

  /** Currently selected file path */
  const selectedFile = ref<string | null>(null)

  /** Loading state */
  const isLoading = ref(false)

  /** Error message */
  const error = ref<string | null>(null)

  // ==========================================================================
  // Computed
  // ==========================================================================

  /** Whether a workspace is currently open */
  const hasWorkspace = computed(() => workspaceRoot.value !== null)

  /** Whether there are any contents to display */
  const hasContents = computed(() => contents.value.length > 0)

  /** Directories in the current path */
  const directories = computed(() =>
    contents.value.filter((entry) => entry.isDir)
  )

  /** Files in the current path */
  const files = computed(() =>
    contents.value.filter((entry) => !entry.isDir)
  )

  /** Sorted contents - directories first, then files, alphabetically */
  const sortedContents = computed(() => {
    const dirs = [...directories.value].sort((a, b) =>
      a.name.localeCompare(b.name)
    )
    const filesSorted = [...files.value].sort((a, b) =>
      a.name.localeCompare(b.name)
    )
    return [...dirs, ...filesSorted]
  })

  /** Whether a folder is expanded */
  const isFolderExpanded = (path: string): boolean => {
    return expandedFolders.value.has(path)
  }

  // ==========================================================================
  // Actions
  // ==========================================================================

  /**
   * Initialize the file explorer by getting the workspace root
   * and loading initial contents.
   */
  async function initialize(): Promise<void> {
    isLoading.value = true
    error.value = null

    try {
      const root = await getWorkspaceRoot()
      workspaceRoot.value = root

      if (root) {
        currentPath.value = root
        await loadContents(root)
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to initialize file explorer'
      console.error('[fileExplorerStore] Initialize error:', err)
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Load the contents of a directory.
   */
  async function loadContents(path: string): Promise<void> {
    try {
      const entries = await listDirectory(path)
      contents.value = entries
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to load directory contents'
      console.error('[fileExplorerStore] Load contents error:', err)
      throw err
    }
  }

  /**
   * Navigate to a directory.
   */
  async function navigateTo(path: string): Promise<void> {
    isLoading.value = true
    error.value = null

    try {
      await loadContents(path)
      currentPath.value = path
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to navigate to directory'
      console.error('[fileExplorerStore] Navigate error:', err)
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Reload the current directory.
   */
  async function refresh(): Promise<void> {
    if (!currentPath.value) {
      return
    }

    isLoading.value = true
    error.value = null

    try {
      await loadContents(currentPath.value)
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to refresh directory'
      console.error('[fileExplorerStore] Refresh error:', err)
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Toggle a folder's expanded state.
   */
  function toggleFolder(path: string): void {
    if (expandedFolders.value.has(path)) {
      expandedFolders.value.delete(path)
    } else {
      expandedFolders.value.add(path)
    }
    // Trigger reactivity by creating a new Set
    expandedFolders.value = new Set(expandedFolders.value)
  }

  /**
   * Expand a folder.
   */
  function expandFolder(path: string): void {
    if (!expandedFolders.value.has(path)) {
      expandedFolders.value.add(path)
      expandedFolders.value = new Set(expandedFolders.value)
    }
  }

  /**
   * Collapse a folder.
   */
  function collapseFolder(path: string): void {
    if (expandedFolders.value.has(path)) {
      expandedFolders.value.delete(path)
      expandedFolders.value = new Set(expandedFolders.value)
    }
  }

  /**
   * Collapse all folders.
   */
  function collapseAll(): void {
    expandedFolders.value = new Set()
  }

  /**
   * Select a file.
   */
  function selectFile(path: string): void {
    selectedFile.value = path
  }

  /**
   * Clear file selection.
   */
  function clearSelection(): void {
    selectedFile.value = null
  }

  /**
   * Open the selected file in the graph view.
   * Uses graphStore.loadFromPythonFile to parse and display the file.
   */
  async function openSelectedFile(): Promise<void> {
    if (!selectedFile.value) {
      return
    }

    isLoading.value = true
    error.value = null

    try {
      const graphStore = useGraphStore()
      await graphStore.loadFromPythonFile(selectedFile.value)
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to open file'
      console.error('[fileExplorerStore] Open file error:', err)
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Open a specific file path in the graph view.
   */
  async function openFile(path: string): Promise<void> {
    selectFile(path)
    await openSelectedFile()
  }

  /**
   * Navigate up to the parent directory.
   */
  async function navigateUp(): Promise<void> {
    if (!currentPath.value || currentPath.value === workspaceRoot.value) {
      return
    }

    // Get parent path
    const parts = currentPath.value.split('/')
    parts.pop()
    const parentPath = parts.join('/') || '/'

    // Don't navigate above workspace root
    if (workspaceRoot.value && !parentPath.startsWith(workspaceRoot.value)) {
      return
    }

    await navigateTo(parentPath)
  }

  /**
   * Set a new workspace root.
   */
  async function setWorkspaceRoot(path: string): Promise<void> {
    workspaceRoot.value = path
    currentPath.value = path
    expandedFolders.value = new Set()
    selectedFile.value = null
    await loadContents(path)
  }

  /**
   * Clear the error state.
   */
  function clearError(): void {
    error.value = null
  }

  /**
   * Reset the store to initial state.
   */
  function reset(): void {
    workspaceRoot.value = null
    currentPath.value = null
    contents.value = []
    expandedFolders.value = new Set()
    selectedFile.value = null
    isLoading.value = false
    error.value = null
  }

  return {
    // State
    workspaceRoot,
    currentPath,
    contents,
    expandedFolders,
    selectedFile,
    isLoading,
    error,

    // Computed
    hasWorkspace,
    hasContents,
    directories,
    files,
    sortedContents,
    isFolderExpanded,

    // Actions
    initialize,
    navigateTo,
    refresh,
    toggleFolder,
    expandFolder,
    collapseFolder,
    collapseAll,
    selectFile,
    clearSelection,
    openSelectedFile,
    openFile,
    navigateUp,
    setWorkspaceRoot,
    clearError,
    reset
  }
})
