/**
 * Recent Files Composable
 *
 * Manages a list of recently opened file paths in localStorage.
 * Most recently opened file appears first. Limited to 10 entries.
 */
import { ref } from 'vue'

const STORAGE_KEY = 'flowforge-recent-files'
const MAX_RECENT_FILES = 10

/** Shared reactive state so all consumers see the same list. */
const recentFiles = ref<string[]>(loadFromStorage())

function loadFromStorage(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        return parsed.filter((p): p is string => typeof p === 'string').slice(0, MAX_RECENT_FILES)
      }
    }
  } catch {
    // Corrupted data – start fresh
  }
  return []
}

function saveToStorage(files: string[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(files))
  } catch {
    // Storage full or unavailable – ignore
  }
}

/**
 * Composable for managing recent files.
 */
export function useRecentFiles() {
  /**
   * Add a file path to the top of the recent files list.
   * Removes duplicates and enforces the max limit.
   */
  function addRecentFile(path: string): void {
    const filtered = recentFiles.value.filter((p) => p !== path)
    filtered.unshift(path)
    recentFiles.value = filtered.slice(0, MAX_RECENT_FILES)
    saveToStorage(recentFiles.value)
  }

  /**
   * Returns the current list of recent file paths.
   */
  function getRecentFiles(): string[] {
    return recentFiles.value
  }

  /**
   * Clears all recent files.
   */
  function clearRecentFiles(): void {
    recentFiles.value = []
    saveToStorage([])
  }

  return {
    recentFiles,
    addRecentFile,
    getRecentFiles,
    clearRecentFiles,
  }
}

export type UseRecentFilesReturn = ReturnType<typeof useRecentFiles>
