#!/usr/bin/env node

import { execSync, spawn } from 'child_process';
import { readFileSync, readdirSync, existsSync } from 'fs';
import { join, basename } from 'path';
import { homedir } from 'os';

const CLAUDE_MEMORY_BASE = join(homedir(), '.claude', 'projects');

function findMemoryDirs() {
  const dirs = [];
  if (!existsSync(CLAUDE_MEMORY_BASE)) return dirs;

  for (const project of readdirSync(CLAUDE_MEMORY_BASE)) {
    const memoryDir = join(CLAUDE_MEMORY_BASE, project, 'memory');
    if (existsSync(memoryDir)) {
      dirs.push({ project, path: memoryDir });
    }
  }
  return dirs;
}

function parseMemoryFile(filePath) {
  try {
    const content = readFileSync(filePath, 'utf-8');
    const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);

    if (!frontmatterMatch) {
      return { name: basename(filePath, '.md'), content, metadata: {} };
    }

    const [, frontmatter, body] = frontmatterMatch;
    const metadata = {};

    for (const line of frontmatter.split('\n')) {
      const colonIdx = line.indexOf(':');
      if (colonIdx > 0) {
        const key = line.slice(0, colonIdx).trim();
        const value = line.slice(colonIdx + 1).trim();
        metadata[key] = value;
      }
    }

    return {
      name: metadata.name || basename(filePath, '.md'),
      description: metadata.description || '',
      type: metadata['metadata']?.type || metadata.type || 'unknown',
      content: body.trim(),
      metadata
    };
  } catch (err) {
    console.error(`Failed to parse ${filePath}: ${err.message}`);
    return null;
  }
}

async function importMemory(memory, project, options = {}) {
  const namespace = options.namespace || `claude-memory-${project}`;
  const key = `${memory.type || 'memory'}-${memory.name}`;
  const value = JSON.stringify({
    name: memory.name,
    description: memory.description,
    type: memory.type,
    content: memory.content,
    source: 'claude-code',
    project,
    importedAt: new Date().toISOString()
  });

  try {
    execSync(
      `npx @claude-flow/cli@latest memory store --key "${key}" --value '${value.replace(/'/g, "\\'")}' --namespace "${namespace}"`,
      { stdio: options.verbose ? 'inherit' : 'pipe', timeout: 30000 }
    );
    return true;
  } catch (err) {
    if (options.verbose) {
      console.error(`Failed to import ${key}: ${err.message}`);
    }
    return false;
  }
}

async function importAll(options = {}) {
  const memoryDirs = findMemoryDirs();

  if (memoryDirs.length === 0) {
    console.log('No Claude Code memory directories found.');
    return { imported: 0, failed: 0, total: 0 };
  }

  console.log(`Found ${memoryDirs.length} project(s) with memories.`);

  let imported = 0;
  let failed = 0;
  let total = 0;

  for (const { project, path } of memoryDirs) {
    const files = readdirSync(path).filter(f => f.endsWith('.md') && f !== 'MEMORY.md');

    if (files.length === 0) continue;

    console.log(`\nProject: ${project} (${files.length} memories)`);

    for (const file of files) {
      total++;
      const memory = parseMemoryFile(join(path, file));

      if (!memory) {
        failed++;
        continue;
      }

      const success = await importMemory(memory, project, options);
      if (success) {
        imported++;
        if (options.verbose) {
          console.log(`  ✓ ${memory.name}`);
        }
      } else {
        failed++;
        if (options.verbose) {
          console.log(`  ✗ ${memory.name}`);
        }
      }
    }
  }

  console.log(`\nImport complete: ${imported}/${total} succeeded, ${failed} failed`);
  return { imported, failed, total };
}

async function importProject(projectPath, options = {}) {
  const memoryDir = join(projectPath, 'memory');

  if (!existsSync(memoryDir)) {
    console.log('No memory directory found in project.');
    return { imported: 0, failed: 0, total: 0 };
  }

  const files = readdirSync(memoryDir).filter(f => f.endsWith('.md') && f !== 'MEMORY.md');
  const project = basename(projectPath);

  console.log(`Importing ${files.length} memories from ${project}...`);

  let imported = 0;
  let failed = 0;

  for (const file of files) {
    const memory = parseMemoryFile(join(memoryDir, file));

    if (!memory) {
      failed++;
      continue;
    }

    const success = await importMemory(memory, project, options);
    if (success) {
      imported++;
    } else {
      failed++;
    }
  }

  console.log(`Import complete: ${imported}/${files.length} succeeded`);
  return { imported, failed, total: files.length };
}

const command = process.argv[2];
const args = process.argv.slice(3);

const options = {
  verbose: args.includes('--verbose') || args.includes('-v'),
  namespace: args.find(a => a.startsWith('--namespace='))?.split('=')[1]
};

switch (command) {
  case 'import-all':
    importAll(options);
    break;

  case 'import':
    const projectPath = args.find(a => !a.startsWith('-')) || process.cwd();
    importProject(projectPath, options);
    break;

  case 'list':
    const dirs = findMemoryDirs();
    if (dirs.length === 0) {
      console.log('No Claude Code memory directories found.');
    } else {
      console.log('Claude Code memory directories:');
      for (const { project, path } of dirs) {
        const count = readdirSync(path).filter(f => f.endsWith('.md') && f !== 'MEMORY.md').length;
        console.log(`  ${project}: ${count} memories`);
      }
    }
    break;

  case 'help':
  default:
    console.log(`
auto-memory-hook.mjs - Import Claude Code memories into AgentDB

Commands:
  import-all              Import all memories from all projects
  import [path]           Import memories from a specific project
  list                    List all Claude Code memory directories
  help                    Show this help message

Options:
  --verbose, -v           Show detailed output
  --namespace=NAME        Override default namespace

Examples:
  node auto-memory-hook.mjs import-all
  node auto-memory-hook.mjs import-all --verbose
  node auto-memory-hook.mjs list
`);
}
