/**
 * Bridge Layer
 *
 * Provides a unified API for communicating with the Tauri backend.
 * Replaces the original ComfyUI fetch-based API.
 */

export { TauriAPI, api } from './api.js';
export { subscribeToEvents, type EventSubscription } from './events.js';
export {
  openFile,
  saveFile,
  readWorkflow,
  writeWorkflow,
  openPythonFile,
  savePythonFileDialog,
  readPythonFile,
  writePythonFile,
  writePythonFileWithBackup,
  fileExists,
  getFileInfo,
  safeWritePythonFile,
  type WriteResult,
  type FileInfo,
} from './files.js';
export { importAsset, getAssetUrl } from './assets.js';
export {
  openInEditor,
  openNodeSource,
  detectEditors,
  setEditorCommand,
  getEditorCommand,
  initEditorBridge,
  cleanupEditorBridge,
  type OpenEditorResponse,
  type EditorInfo,
} from './editor.js';
// Code generation bridge
export {
  // Primary API
  generateCode,
  validateCode,
  generateDiff,
  applyChanges,
  // Additional functions
  generateDiffFromStrings,
  applyContent,
  // Legacy API (backwards compatible)
  generatePython,
  validatePython,
  isValidPython,
  generateAndSave,
  loadAndValidate,
  validateBatch,
  // Helper utilities
  ValidationHelpers,
  GenerationHelpers,
  DiffHelpers,
  // Types
  type Severity,
  type ValidationError,
  type ValidationResult,
  type ImportInfo,
  type GeneratedCode,
  type ValidationOptions,
  type GenerationOptions,
  type DiffLineType,
  type DiffLine,
  type DiffHunk,
  type DiffStats,
  type DiffResult,
  type SideBySideLine,
  type SideBySideDiff,
  type DiffOptions,
  type ApplyResult,
  type GenerationResult,
} from './codegen.js';

// Trinity introspection bridge
export {
  checkTrinityStatus,
  getRegistryContents,
  queryInstances,
  getRecentEvents,
  filterEntriesByType,
  getRegistrySummary,
  inspectType,
  inspectorGet,
  type TrinityStatus,
  type RegistryEntry,
  type RegistryEntryType,
  type RegistryContents,
  type TrinityInstance,
  type InstancesQueryResult,
  type TrinityEvent,
  type EventSeverity,
  type RecentEventsResult,
  type InspectionResult,
  type HierarchyEntry,
  type DecoratorEntry,
} from './trinity.js';
