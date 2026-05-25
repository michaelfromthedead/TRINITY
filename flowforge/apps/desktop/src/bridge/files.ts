/**
 * File Dialog Bridge
 *
 * Provides native file dialog integration via Tauri.
 */

import { invoke } from '@tauri-apps/api/core';
import type { WorkflowSchema } from '@flowforge/core';
import { isTauri } from '@/services';

// ==========================================================================
// Types
// ==========================================================================

/**
 * Result of a file write operation with backup support.
 */
export interface WriteResult {
  success: boolean;
  path: string;
  backupPath?: string | undefined;
  error?: string | undefined;
}

/**
 * Entry representing a file or directory in a listing.
 */
export interface FileEntry {
  name: string;
  path: string;
  isDir: boolean;
  size: number;
  modified: number | null; // Unix timestamp
}

/**
 * File information from the filesystem.
 */
export interface FileInfo {
  path: string;
  exists: boolean;
  size?: number | undefined;
  modified?: number | undefined;
  isReadonly: boolean;
}

/**
 * File filter for dialogs.
 */
export interface FileFilter {
  name: string;
  extensions: string[];
}

/**
 * Default workflow file filters.
 */
export const WORKFLOW_FILTERS: FileFilter[] = [
  { name: 'FlowForge Workflow', extensions: ['flowforge', 'json'] },
  { name: 'All Files', extensions: ['*'] },
];

/**
 * Python file filters for Trinity integration.
 */
export const PYTHON_FILTERS: FileFilter[] = [
  { name: 'Python Files', extensions: ['py'] },
  { name: 'All Files', extensions: ['*'] },
];

/**
 * Open file dialog.
 */
export async function openFile(options?: {
  filters?: FileFilter[];
  title?: string;
  defaultPath?: string;
}): Promise<string | null> {
  return await invoke<string | null>('open_file_dialog', {
    request: {
      filters: options?.filters ?? WORKFLOW_FILTERS,
      title: options?.title,
      defaultPath: options?.defaultPath,
    },
  });
}

/**
 * Save file dialog.
 */
export async function saveFile(options?: {
  filters?: FileFilter[];
  title?: string;
  defaultPath?: string;
}): Promise<string | null> {
  return await invoke<string | null>('save_file_dialog', {
    request: {
      filters: options?.filters ?? WORKFLOW_FILTERS,
      title: options?.title,
      defaultPath: options?.defaultPath,
    },
  });
}

/**
 * Read a workflow file.
 */
export async function readWorkflow(path: string): Promise<{
  path: string;
  content: WorkflowSchema;
}> {
  const result = await invoke<{ path: string; content: unknown }>('read_workflow_file', {
    path,
  });

  return {
    path: result.path,
    content: result.content as WorkflowSchema,
  };
}

/**
 * Write a workflow file.
 */
export async function writeWorkflow(path: string, workflow: WorkflowSchema): Promise<boolean> {
  return await invoke<boolean>('write_workflow_file', {
    path,
    content: workflow,
  });
}

/**
 * Open a workflow (combines dialog and read).
 */
export async function openWorkflow(): Promise<{
  path: string;
  content: WorkflowSchema;
} | null> {
  const path = await openFile({
    title: 'Open Workflow',
  });

  if (path === null) {
    return null;
  }

  return await readWorkflow(path);
}

/**
 * Save a workflow (combines dialog and write).
 */
export async function saveWorkflowAs(workflow: WorkflowSchema): Promise<string | null> {
  const path = await saveFile({
    title: 'Save Workflow',
  });

  if (path === null) {
    return null;
  }

  await writeWorkflow(path, workflow);
  return path;
}

// ==========================================================================
// Python File Operations (Trinity Integration)
// ==========================================================================

/**
 * Open Python file dialog.
 */
export async function openPythonFile(options?: {
  title?: string;
  defaultPath?: string;
}): Promise<string | null> {
  return await invoke<string | null>('open_file_dialog', {
    request: {
      filters: PYTHON_FILTERS,
      title: options?.title ?? 'Open Python File',
      defaultPath: options?.defaultPath,
    },
  });
}

/**
 * Save Python file dialog.
 */
export async function savePythonFileDialog(options?: {
  title?: string;
  defaultPath?: string;
}): Promise<string | null> {
  return await invoke<string | null>('save_file_dialog', {
    request: {
      filters: PYTHON_FILTERS,
      title: options?.title ?? 'Save Python File',
      defaultPath: options?.defaultPath,
    },
  });
}

/**
 * Read a Python file's content.
 */
export async function readPythonFile(path: string): Promise<{
  path: string;
  content: string;
}> {
  return await invoke<{ path: string; content: string }>('read_python_file', {
    path,
  });
}

/**
 * Write content to a Python file.
 */
export async function writePythonFile(path: string, content: string): Promise<boolean> {
  return await invoke<boolean>('write_python_file', {
    path,
    content,
  });
}

/**
 * Open a Python file (combines dialog and read).
 */
export async function openPython(): Promise<{
  path: string;
  content: string;
} | null> {
  const path = await openPythonFile({
    title: 'Open Python File',
  });

  if (path === null) {
    return null;
  }

  return await readPythonFile(path);
}

/**
 * Save Python content with dialog (combines dialog and write).
 */
export async function savePythonAs(content: string): Promise<string | null> {
  const path = await savePythonFileDialog({
    title: 'Save Python File',
  });

  if (path === null) {
    return null;
  }

  await writePythonFile(path, content);
  return path;
}

// ==========================================================================
// Enhanced File Operations
// ==========================================================================

/**
 * Write a Python file with automatic backup creation.
 * Creates a .bak file before overwriting existing files.
 *
 * @param path - Path to the file
 * @param content - Content to write
 * @returns Write result with backup information
 */
export async function writePythonFileWithBackup(
  path: string,
  content: string
): Promise<WriteResult> {
  try {
    const result = await invoke<{
      success: boolean;
      path: string;
      backup_path?: string;
      error?: string;
    }>('write_text_file_with_backup', { path, content });

    return {
      success: result.success,
      path: result.path,
      backupPath: result.backup_path,
      error: result.error,
    };
  } catch (error) {
    return {
      success: false,
      path,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Check if a file exists.
 *
 * @param path - Path to check
 * @returns True if file exists
 */
export async function fileExists(path: string): Promise<boolean> {
  return await invoke<boolean>('file_exists', { path });
}

/**
 * Get information about a file.
 *
 * @param path - Path to the file
 * @returns File information
 */
export async function getFileInfo(path: string): Promise<FileInfo> {
  const result = await invoke<{
    path: string;
    exists: boolean;
    size?: number;
    modified?: number;
    is_readonly: boolean;
  }>('get_file_info', { path });

  return {
    path: result.path,
    exists: result.exists,
    size: result.size,
    modified: result.modified,
    isReadonly: result.is_readonly,
  };
}

/**
 * Safe write operation that verifies the file was written correctly.
 * Compares file size after write to ensure content was saved.
 *
 * @param path - Path to the file
 * @param content - Content to write
 * @returns Write result
 */
export async function safeWritePythonFile(
  path: string,
  content: string
): Promise<WriteResult> {
  // Write with backup
  const result = await writePythonFileWithBackup(path, content);

  if (!result.success) {
    return result;
  }

  // Verify the write by checking file info
  try {
    const info = await getFileInfo(path);

    if (!info.exists) {
      return {
        success: false,
        path,
        error: 'File was not created after write',
      };
    }

    // Check if file size is reasonable (at least as long as content in bytes)
    const expectedMinSize = new TextEncoder().encode(content).length;
    if (info.size !== undefined && info.size < expectedMinSize * 0.9) {
      return {
        success: false,
        path,
        error: 'File size mismatch after write - content may be truncated',
        backupPath: result.backupPath,
      };
    }

    return result;
  } catch (error) {
    // If verification fails, still return success since write succeeded
    console.warn('Write verification failed:', error);
    return result;
  }
}

// ==========================================================================
// Directory Listing Operations
// ==========================================================================

/**
 * List the contents of a directory.
 *
 * @param path - Path to the directory to list
 * @returns Array of file entries in the directory
 */
export async function listDirectory(path: string): Promise<FileEntry[]> {
  if (!isTauri()) {
    console.warn('[Files] listDirectory not available in browser mode');
    return [];
  }

  try {
    const result = await invoke<
      Array<{
        name: string;
        path: string;
        is_dir: boolean;
        size: number;
        modified: number | null;
      }>
    >('list_directory', { path });

    return result.map((entry) => ({
      name: entry.name,
      path: entry.path,
      isDir: entry.is_dir,
      size: entry.size,
      modified: entry.modified,
    }));
  } catch (error) {
    throw new Error(
      `Failed to list directory: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Get the workspace root path.
 *
 * @returns The workspace root path, or null if not set
 */
export async function getWorkspaceRoot(): Promise<string | null> {
  if (!isTauri()) {
    console.warn('[Files] getWorkspaceRoot not available in browser mode');
    return null;
  }

  try {
    return await invoke<string | null>('get_workspace_root');
  } catch (error) {
    throw new Error(
      `Failed to get workspace root: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}
