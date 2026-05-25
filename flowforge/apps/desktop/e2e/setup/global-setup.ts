import { FullConfig } from '@playwright/test'
import { spawn, ChildProcess } from 'child_process'
import { resolve, dirname } from 'path'
import { existsSync } from 'fs'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

/**
 * Global setup for Playwright E2E tests with Tauri.
 *
 * This setup:
 * 1. Verifies the Tauri binary exists
 * 2. Optionally builds the app if TAURI_BUILD=true
 * 3. Stores the binary path for test use
 */

let buildProcess: ChildProcess | null = null

async function globalSetup(config: FullConfig): Promise<void> {
  console.log('\n[E2E Setup] Starting global setup...')

  const rootDir = resolve(__dirname, '../..')
  const srcTauriDir = resolve(rootDir, 'src-tauri')

  // Check if we should build the app
  const shouldBuild = process.env.TAURI_BUILD === 'true'

  if (shouldBuild) {
    console.log('[E2E Setup] Building Tauri app (TAURI_BUILD=true)...')

    await new Promise<void>((resolve, reject) => {
      buildProcess = spawn('npm', ['run', 'tauri:build'], {
        cwd: rootDir,
        stdio: 'inherit',
        shell: true,
      })

      buildProcess.on('close', (code) => {
        if (code === 0) {
          console.log('[E2E Setup] Build completed successfully')
          resolve()
        } else {
          reject(new Error(`Build failed with code ${code}`))
        }
      })

      buildProcess.on('error', (err) => {
        reject(err)
      })
    })
  }

  // Verify binary exists
  const platform = process.platform
  const binaryName = platform === 'win32' ? 'flowforge.exe' : 'flowforge'
  const targetDir = resolve(srcTauriDir, 'target')

  const debugBinary = resolve(targetDir, 'debug', binaryName)
  const releaseBinary = resolve(targetDir, 'release', binaryName)

  const binaryPath = process.env.TAURI_BINARY_PATH
    ?? (existsSync(debugBinary) ? debugBinary : releaseBinary)

  if (!existsSync(binaryPath)) {
    console.warn(
      `[E2E Setup] Warning: Tauri binary not found at expected paths:\n` +
      `  - Debug: ${debugBinary}\n` +
      `  - Release: ${releaseBinary}\n` +
      `\nTo build the app, run: npm run tauri:build\n` +
      `Or set TAURI_BUILD=true to build before tests.\n` +
      `\nTests will attempt to use dev server mode instead.`
    )

    // Store flag for dev mode
    process.env.TAURI_DEV_MODE = 'true'
  } else {
    console.log(`[E2E Setup] Using Tauri binary: ${binaryPath}`)
    process.env.TAURI_BINARY_PATH = binaryPath
    process.env.TAURI_DEV_MODE = 'false'
  }

  // Create fixtures directory if needed
  const fixturesDir = resolve(__dirname, '../fixtures')
  if (!existsSync(fixturesDir)) {
    const { mkdirSync } = await import('fs')
    mkdirSync(fixturesDir, { recursive: true })
  }

  console.log('[E2E Setup] Global setup complete\n')
}

export default globalSetup
