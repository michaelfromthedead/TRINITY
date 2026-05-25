/**
 * FlowForge Prettier Configuration
 *
 * Consistent code formatting across the entire monorepo.
 */
export default {
  // Line length and wrapping
  printWidth: 100,
  tabWidth: 2,
  useTabs: false,

  // Semicolons and quotes
  semi: true,
  singleQuote: true,
  quoteProps: 'as-needed',
  jsxSingleQuote: false,

  // Trailing commas and brackets
  trailingComma: 'es5',
  bracketSpacing: true,
  bracketSameLine: false,

  // Arrow functions
  arrowParens: 'always',

  // End of line
  endOfLine: 'lf',

  // Prose wrapping for markdown
  proseWrap: 'preserve',

  // HTML/Vue specific
  htmlWhitespaceSensitivity: 'css',
  vueIndentScriptAndStyle: false,

  // Embedded languages
  embeddedLanguageFormatting: 'auto',

  // Experimental features
  experimentalTernaries: false,

  // Plugin order
  plugins: [],

  // Override for specific file types
  overrides: [
    {
      files: '*.json',
      options: {
        printWidth: 80,
        tabWidth: 2,
      },
    },
    {
      files: '*.md',
      options: {
        proseWrap: 'always',
        printWidth: 80,
      },
    },
    {
      files: ['*.yaml', '*.yml'],
      options: {
        tabWidth: 2,
        singleQuote: false,
      },
    },
    {
      files: '*.vue',
      options: {
        singleQuote: true,
        htmlWhitespaceSensitivity: 'ignore',
      },
    },
  ],
};
