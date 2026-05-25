/**
 * Asset Management Bridge
 *
 * Handles importing and accessing assets via Tauri.
 */

import { invoke } from '@tauri-apps/api/core';
import { convertFileSrc } from '@tauri-apps/api/core';

/**
 * Asset import result.
 */
export interface ImportAssetResult {
  id: string;
  localPath: string;
  assetType: string;
}

/**
 * Import an asset into the project.
 */
export async function importAsset(
  sourcePath: string,
  assetType?: string
): Promise<ImportAssetResult> {
  return await invoke<ImportAssetResult>('import_asset', {
    request: {
      sourcePath,
      assetType,
    },
  });
}

/**
 * Get a URL for accessing an asset.
 * Converts a local file path to a Tauri asset protocol URL.
 */
export function getAssetUrl(localPath: string): string {
  return convertFileSrc(localPath);
}

/**
 * Get asset URL via backend (for validation).
 */
export async function getAssetUrlValidated(localPath: string): Promise<string> {
  return await invoke<string>('get_asset_url', { localPath });
}
