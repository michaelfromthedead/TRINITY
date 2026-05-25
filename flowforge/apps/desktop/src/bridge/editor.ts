/**
 * Editor Integration Bridge
 *
 * Provides functions to open source files in external editors.
 * Supports custom editor commands with line number jumping.
 */

import { invoke } from '@tauri-apps/api/core';

/**
 * Response from the open_in_editor command.
 */
export interface OpenEditorResponse {
  success: boolean;
  message: string;
}

/**
 * Information about a detected editor.
 */
export interface EditorInfo {
  /** Display name of the editor */
  name: string;
  /** Command template (use {file} and {line} as placeholders) */
  command: string;
  /** Whether the editor was detected on the system */
  detected: boolean;
}

/**
 * Settings store reference for editor command.
 * This will be populated by the settings system.
 */
let editorCommand: string | null = null;

/**
 * Set the editor command to use for opening files.
 * Call this when settings are loaded or changed.
 *
 * @param command - The editor command template (e.g., "code --goto {file}:{line}")
 */
export function setEditorCommand(command: string | null): void {
  editorCommand = command;
}

/**
 * Get the currently configured editor command.
 */
export function getEditorCommand(): string | null {
  return editorCommand;
}

/**
 * Open a file in an external editor.
 *
 * @param file - The path to the file to open
 * @param line - Optional line number to jump to
 * @returns Promise resolving to the response from the backend
 *
 * @example
 * ```typescript
 * // Open file at line 42
 * await openInEditor('/path/to/file.py', 42);
 *
 * // Open file (no specific line)
 * await openInEditor('/path/to/file.py');
 * ```
 */
export async function openInEditor(
  file: string,
  line?: number
): Promise<OpenEditorResponse> {
  try {
    const response = await invoke<OpenEditorResponse>('open_in_editor', {
      request: {
        file,
        line: line ?? null,
        editor_command: editorCommand,
      },
    });
    return response;
  } catch (error) {
    console.error('[Editor] Failed to open file in editor:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Detect available editors on the system.
 *
 * @returns Promise resolving to a list of detected editors
 */
export async function detectEditors(): Promise<EditorInfo[]> {
  try {
    return await invoke<EditorInfo[]>('detect_editors');
  } catch (error) {
    console.error('[Editor] Failed to detect editors:', error);
    return [
      {
        name: 'System Default',
        command: '',
        detected: true,
      },
    ];
  }
}

/**
 * Open a source file from a Trinity node.
 * This is a convenience function that extracts source info from node data.
 *
 * @param source - Source information object with file and line
 * @returns Promise resolving when the file is opened (or error message)
 */
export async function openNodeSource(source: {
  file: string;
  line?: number;
}): Promise<OpenEditorResponse> {
  if (!source.file) {
    return {
      success: false,
      message: 'No source file information available',
    };
  }
  return openInEditor(source.file, source.line);
}

/**
 * Event handler for the 'flowforge:navigate-to-source' custom event.
 * This event is dispatched by nodes when the user wants to navigate to source.
 *
 * @param event - The custom event with file/line detail
 */
export function handleNavigateToSource(
  event: CustomEvent<{ file: string; line?: number }>
): void {
  const { file, line } = event.detail;
  openInEditor(file, line).then((response) => {
    if (!response.success) {
      console.warn('[Editor] Failed to navigate to source:', response.message);
    }
  });
}

/**
 * Initialize the editor bridge.
 * Sets up the global event listener for navigate-to-source events.
 */
export function initEditorBridge(): void {
  window.addEventListener(
    'flowforge:navigate-to-source',
    handleNavigateToSource as EventListener
  );
  console.log('[Editor] Editor bridge initialized');
}

/**
 * Cleanup the editor bridge.
 * Removes the global event listener.
 */
export function cleanupEditorBridge(): void {
  window.removeEventListener(
    'flowforge:navigate-to-source',
    handleNavigateToSource as EventListener
  );
  console.log('[Editor] Editor bridge cleaned up');
}
