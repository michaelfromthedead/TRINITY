/**
 * Composables Index
 *
 * Export all composables from a single entry point.
 */

export { useFileOperations } from './useFileOperations'
export type { FileOperationsOptions, UseFileOperationsReturn } from './useFileOperations'

export { useWindowTitle } from './useWindowTitle'
export type { WindowTitleOptions, UseWindowTitleReturn } from './useWindowTitle'

export { useTypeFilter, TRINITY_TYPES } from './useTypeFilter'
export type { UseTypeFilterReturn, FilterableTrinityType } from './useTypeFilter'

export { useNodeSearch } from './useNodeSearch'
export type {
  UseNodeSearchOptions,
  UseNodeSearchReturn,
  NodeTypeFilter,
  SearchResult
} from './useNodeSearch'

export {
  useSourceNavigation,
  dispatchNavigateToSource,
  dispatchClearHighlight,
  NAVIGATE_TO_SOURCE_EVENT,
  EXTERNAL_NAVIGATION_EVENT,
  CLEAR_HIGHLIGHT_EVENT,
} from './useSourceNavigation'
export type {
  SourceLocation,
  NavigateToSourceEventDetail,
  SourceNavigationOptions,
  UseSourceNavigationReturn,
} from './useSourceNavigation'

export { useEventLog } from './useEventLog'
export type {
  EventLogEntry,
  EventLogFilter,
  UseEventLogOptions,
  UseEventLogReturn
} from './useEventLog'

export { useRegistryPanel } from './useRegistryPanel'
export type {
  RegistryEntry,
  RegistrationStatus,
  UseRegistryPanelOptions,
  UseRegistryPanelReturn
} from './useRegistryPanel'

export { useInstances } from './useInstances'
export type {
  TrinityInstance,
  InstanceGroup,
  InstanceTreeNode,
  ConnectionStatus,
  UseInstancesReturn
} from './useInstances'

export { useInspector } from './useInspector'
export type {
  UseInspectorOptions,
  UseInspectorReturn
} from './useInspector'

export { useEventHighlight } from './useEventHighlight'
export type {
  UseEventHighlightOptions,
  UseEventHighlightReturn
} from './useEventHighlight'

export { useUndoRedo } from './useUndoRedo'
export type {
  HistoryEntry,
  UseUndoRedoOptions,
  UseUndoRedoReturn
} from './useUndoRedo'

export {
  useInlineEdit,
  validatePythonIdentifier,
  validateFieldType,
  validateDefaultValue,
  PYTHON_BUILTIN_TYPES,
  TYPING_TYPES,
  TRINITY_DATA_TYPES,
  ALL_RECOGNIZED_TYPES
} from './useInlineEdit'
export type {
  InlineEditType,
  InlineEditState,
  ValidationResult,
  EditCommitEvent,
  UseInlineEditOptions,
  UseInlineEditReturn
} from './useInlineEdit'

export { useNodeEditing } from './useNodeEditing'
export type {
  TrinityNodeType,
  NodeField,
  NewNodeData,
  PendingChange,
  FieldUpdates,
  NodeEditEvent,
  UseNodeEditingOptions,
  UseNodeEditingReturn
} from './useNodeEditing'

export { useDiffPreview } from './useDiffPreview'
export type {
  DiffLineType,
  DiffLine,
  DiffHunk,
  DiffStats,
  DiffResult,
  SideBySideLine,
  SideBySideDiff,
  DiffViewMode,
  ApplyResult,
  UseDiffPreviewOptions,
  UseDiffPreviewReturn,
} from './useDiffPreview'

export { useKeyboardShortcuts } from './useKeyboardShortcuts'
export type {
  UseKeyboardShortcutsOptions,
  UseKeyboardShortcutsReturn
} from './useKeyboardShortcuts'

export { useFileWatcher } from './useFileWatcher'
export type {
  FileWatcherOptions,
  UseFileWatcherReturn
} from './useFileWatcher'

export { useRecentFiles } from './useRecentFiles'
export type { UseRecentFilesReturn } from './useRecentFiles'

export { useFileConflict } from './useFileConflict'
export type {
  FileConflictOptions,
  FileConflictAction,
  UseFileConflictReturn
} from './useFileConflict'
