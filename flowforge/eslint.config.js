import baseConfig from './config/eslint.base.js';

/**
 * Root ESLint configuration for FlowForge monorepo.
 * Individual packages extend this with their own configs.
 */
export default [
  ...baseConfig,
  {
    ignores: [
      'apps/*/dist/**',
      'packages/*/dist/**',
      '**/node_modules/**',
      'apps/desktop/src-tauri/target/**',
    ],
  },
];
