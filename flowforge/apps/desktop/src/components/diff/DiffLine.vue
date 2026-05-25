<template>
  <div
    :class="[
      'diff-line',
      `diff-line--${type}`,
      { 'diff-line--highlighted': highlighted }
    ]"
  >
    <!-- Line number gutter -->
    <div class="diff-line__gutter">
      <span
        v-if="showOriginalLineNumber"
        class="diff-line__number diff-line__number--original"
      >
        {{ originalLineNumber ?? '' }}
      </span>
      <span
        v-if="showModifiedLineNumber"
        class="diff-line__number diff-line__number--modified"
      >
        {{ modifiedLineNumber ?? '' }}
      </span>
    </div>

    <!-- Change indicator -->
    <div class="diff-line__indicator">
      <span v-if="type === 'added'">+</span>
      <span v-else-if="type === 'removed'">-</span>
      <span v-else-if="type === 'header'">@@</span>
      <span v-else>&nbsp;</span>
    </div>

    <!-- Line content with syntax highlighting -->
    <div class="diff-line__content">
      <code v-if="highlightSyntax" v-html="highlightedContent"></code>
      <code v-else>{{ content }}</code>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { DiffLineType } from '@/composables/useDiffPreview'

// =============================================================================
// PROPS
// =============================================================================

interface Props {
  /** Type of diff line */
  type: DiffLineType
  /** Line content */
  content: string
  /** Line number in original file */
  originalLineNumber?: number | null
  /** Line number in modified file */
  modifiedLineNumber?: number | null
  /** Whether to show the original line number column */
  showOriginalLineNumber?: boolean
  /** Whether to show the modified line number column */
  showModifiedLineNumber?: boolean
  /** Whether line is highlighted (e.g., for search) */
  highlighted?: boolean
  /** Whether to apply syntax highlighting */
  highlightSyntax?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  originalLineNumber: null,
  modifiedLineNumber: null,
  showOriginalLineNumber: true,
  showModifiedLineNumber: true,
  highlighted: false,
  highlightSyntax: true,
})

// =============================================================================
// COMPUTED
// =============================================================================

/**
 * Apply basic Python syntax highlighting.
 * For a production app, consider using a library like highlight.js or Prism.
 */
const highlightedContent = computed(() => {
  if (!props.highlightSyntax || props.type === 'header') {
    return escapeHtml(props.content)
  }

  let html = escapeHtml(props.content)

  // Python keywords
  const keywords = [
    'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while',
    'try', 'except', 'finally', 'with', 'as', 'import', 'from',
    'raise', 'pass', 'break', 'continue', 'and', 'or', 'not', 'in',
    'is', 'None', 'True', 'False', 'self', 'lambda', 'yield', 'async', 'await',
  ]

  // Highlight keywords (word boundaries)
  for (const keyword of keywords) {
    const regex = new RegExp(`\\b(${keyword})\\b`, 'g')
    html = html.replace(regex, '<span class="syntax-keyword">$1</span>')
  }

  // Highlight decorators
  html = html.replace(
    /(@[\w.]+)/g,
    '<span class="syntax-decorator">$1</span>'
  )

  // Highlight strings (single and double quotes)
  html = html.replace(
    /(["'])(?:(?=(\\?))\2.)*?\1/g,
    '<span class="syntax-string">$&</span>'
  )

  // Highlight comments
  html = html.replace(
    /(#.*)$/gm,
    '<span class="syntax-comment">$1</span>'
  )

  // Highlight numbers
  html = html.replace(
    /\b(\d+\.?\d*)\b/g,
    '<span class="syntax-number">$1</span>'
  )

  // Highlight function/class names after def/class
  html = html.replace(
    /(<span class="syntax-keyword">(?:def|class)<\/span>\s+)([\w_]+)/g,
    '$1<span class="syntax-function">$2</span>'
  )

  // Highlight type annotations
  html = html.replace(
    /(:\s*)([\w\[\], ]+)(\s*[=)]|$)/g,
    '$1<span class="syntax-type">$2</span>$3'
  )

  return html
})

/**
 * Escape HTML special characters.
 */
function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  }
  return text.replace(/[&<>"']/g, (m) => map[m] || m)
}
</script>

<style scoped>
.diff-line {
  display: flex;
  font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre;
  min-height: 22px;
}

.diff-line--added {
  background-color: rgba(46, 160, 67, 0.15);
}

.diff-line--removed {
  background-color: rgba(248, 81, 73, 0.15);
}

.diff-line--context,
.diff-line--unchanged {
  background-color: transparent;
}

.diff-line--header {
  background-color: rgba(56, 139, 253, 0.15);
  color: var(--flowforge-text-muted, #8b949e);
  font-style: italic;
}

.diff-line--empty {
  background-color: rgba(128, 128, 128, 0.05);
}

.diff-line--highlighted {
  outline: 2px solid var(--flowforge-accent, #58a6ff);
  outline-offset: -2px;
}

.diff-line__gutter {
  display: flex;
  flex-shrink: 0;
  user-select: none;
}

.diff-line__number {
  display: inline-block;
  min-width: 40px;
  padding: 0 8px;
  text-align: right;
  color: var(--flowforge-text-muted, #6e7681);
  background-color: rgba(0, 0, 0, 0.1);
  border-right: 1px solid var(--flowforge-border, #30363d);
}

.diff-line__number--original {
  background-color: rgba(248, 81, 73, 0.05);
}

.diff-line__number--modified {
  background-color: rgba(46, 160, 67, 0.05);
}

.diff-line--added .diff-line__number--original {
  background-color: transparent;
}

.diff-line--removed .diff-line__number--modified {
  background-color: transparent;
}

.diff-line__indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  flex-shrink: 0;
  font-weight: bold;
  user-select: none;
}

.diff-line--added .diff-line__indicator {
  color: #3fb950;
}

.diff-line--removed .diff-line__indicator {
  color: #f85149;
}

.diff-line--header .diff-line__indicator {
  color: #58a6ff;
}

.diff-line__content {
  flex: 1;
  padding-left: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diff-line__content code {
  font-family: inherit;
  font-size: inherit;
  background: none;
  padding: 0;
  color: var(--flowforge-text, #c9d1d9);
}

/* Syntax highlighting colors */
:deep(.syntax-keyword) {
  color: #ff7b72;
  font-weight: 500;
}

:deep(.syntax-decorator) {
  color: #d2a8ff;
}

:deep(.syntax-string) {
  color: #a5d6ff;
}

:deep(.syntax-comment) {
  color: #8b949e;
  font-style: italic;
}

:deep(.syntax-number) {
  color: #79c0ff;
}

:deep(.syntax-function) {
  color: #d2a8ff;
  font-weight: 500;
}

:deep(.syntax-type) {
  color: #7ee787;
}
</style>
