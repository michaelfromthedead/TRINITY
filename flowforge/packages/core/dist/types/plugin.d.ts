/**
 * Plugin System Types
 *
 * Defines the structure of FlowForge plugins, including manifests,
 * lifecycle hooks, and permission requirements.
 */
import type { SemanticVersion } from './primitives.js';
import type { NodeDefinition } from './node.js';
import type { TypeDefinition } from './primitives.js';
/**
 * Plugin author information.
 */
export interface PluginAuthor {
    readonly name: string;
    readonly email?: string;
    readonly url?: string;
}
/**
 * Plugin dependency specification.
 */
export interface PluginDependency {
    /** Plugin package name */
    readonly name: string;
    /** Version range (semver) */
    readonly version: string;
    /** Whether this dependency is optional */
    readonly optional?: boolean;
}
/**
 * Plugin capability requirements.
 */
export interface PluginCapabilities {
    /** Required filesystem access paths */
    readonly filesystem?: {
        readonly read?: readonly string[];
        readonly write?: readonly string[];
    };
    /** Required network access */
    readonly network?: {
        readonly hosts?: readonly string[];
        readonly protocols?: readonly ('http' | 'https' | 'ws' | 'wss')[];
    };
    /** Required system capabilities */
    readonly system?: {
        readonly clipboard?: boolean;
        readonly notifications?: boolean;
        readonly shell?: boolean;
    };
    /** Custom capabilities */
    readonly custom?: Readonly<Record<string, unknown>>;
}
/**
 * Plugin manifest file structure (plugin.json).
 */
export interface PluginManifest {
    /** Plugin name (unique identifier) */
    readonly name: string;
    /** Display name for UI */
    readonly displayName?: string;
    /** Plugin version */
    readonly version: SemanticVersion;
    /** Plugin description */
    readonly description?: string;
    /** Plugin author(s) */
    readonly author?: PluginAuthor | readonly PluginAuthor[];
    /** Plugin license */
    readonly license?: string;
    /** Repository URL */
    readonly repository?: string;
    /** Homepage URL */
    readonly homepage?: string;
    /** Keywords for search */
    readonly keywords?: readonly string[];
    /** Main entry point (relative path) */
    readonly main: string;
    /** Node types provided by this plugin */
    readonly nodes?: readonly string[];
    /** Custom types defined by this plugin */
    readonly types?: readonly string[];
    /** Plugin dependencies */
    readonly dependencies?: Readonly<Record<string, string>>;
    /** Plugin peer dependencies */
    readonly peerDependencies?: Readonly<Record<string, string>>;
    /** FlowForge SDK version requirement */
    readonly flowforge?: string;
    /** Required capabilities/permissions */
    readonly capabilities?: PluginCapabilities;
    /** Plugin icon (relative path or URL) */
    readonly icon?: string;
    /** Plugin banner image */
    readonly banner?: string;
    /** Minimum supported FlowForge version */
    readonly minVersion?: SemanticVersion;
    /** Maximum supported FlowForge version */
    readonly maxVersion?: SemanticVersion;
    /** Whether this plugin is deprecated */
    readonly deprecated?: boolean;
    /** Replacement plugin if deprecated */
    readonly replacedBy?: string;
    /** Additional metadata */
    readonly extra?: Readonly<Record<string, unknown>>;
}
/**
 * Custom type registration.
 */
export interface CustomTypeDefinition extends TypeDefinition {
    /** Type display name */
    readonly displayName?: string;
    /** Type description */
    readonly description?: string;
    /** Type color for canvas rendering */
    readonly color: string;
    /** Validation function (as string, evaluated at runtime) */
    readonly validate?: string;
    /** Coercion function (as string) */
    readonly coerce?: string;
    /** Default value */
    readonly defaultValue?: unknown;
}
/**
 * Plugin context passed to lifecycle hooks.
 */
export interface PluginContext {
    /** Plugin manifest */
    readonly manifest: PluginManifest;
    /** Plugin installation directory */
    readonly pluginDir: string;
    /** FlowForge app data directory */
    readonly appDataDir: string;
    /** Plugin-specific data directory */
    readonly dataDir: string;
    /** Plugin-specific cache directory */
    readonly cacheDir: string;
    /** Log a message */
    log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: unknown): void;
    /** Store plugin configuration */
    setConfig<T>(key: string, value: T): Promise<void>;
    /** Retrieve plugin configuration */
    getConfig<T>(key: string): Promise<T | undefined>;
    /** Check if a capability is granted */
    hasCapability(capability: keyof PluginCapabilities | string): boolean;
}
/**
 * Plugin definition exported from main entry point.
 */
export interface PluginDefinition {
    /** Plugin name (must match manifest) */
    readonly name: string;
    /** Node definitions */
    readonly nodes: readonly NodeDefinition[];
    /** Custom type definitions */
    readonly types?: Readonly<Record<string, CustomTypeDefinition>>;
    /** Called when plugin is loaded */
    onLoad?(context: PluginContext): void | Promise<void>;
    /** Called when plugin is unloaded */
    onUnload?(context: PluginContext): void | Promise<void>;
    /** Called when plugin is enabled */
    onEnable?(context: PluginContext): void | Promise<void>;
    /** Called when plugin is disabled */
    onDisable?(context: PluginContext): void | Promise<void>;
    /** Called when FlowForge starts with this plugin */
    onStartup?(context: PluginContext): void | Promise<void>;
    /** Called before FlowForge shuts down */
    onShutdown?(context: PluginContext): void | Promise<void>;
}
/**
 * Loaded plugin instance.
 */
export interface LoadedPlugin {
    /** Plugin manifest */
    readonly manifest: PluginManifest;
    /** Plugin definition */
    readonly definition: PluginDefinition;
    /** Plugin status */
    readonly status: PluginStatus;
    /** When the plugin was loaded */
    readonly loadedAt: string;
    /** Plugin context */
    readonly context: PluginContext;
}
/**
 * Plugin status.
 */
export type PluginStatus = 'loading' | 'active' | 'disabled' | 'error' | 'outdated';
/**
 * Plugin load error.
 */
export interface PluginLoadError {
    readonly pluginName: string;
    readonly code: string;
    readonly message: string;
    readonly cause?: Error;
}
/**
 * Plugin discovery result.
 */
export interface DiscoveredPlugin {
    /** Manifest file path */
    readonly manifestPath: string;
    /** Plugin manifest */
    readonly manifest: PluginManifest;
    /** Discovery source */
    readonly source: 'builtin' | 'user' | 'project';
    /** Whether this plugin is valid */
    readonly valid: boolean;
    /** Validation errors if invalid */
    readonly errors?: readonly string[];
}
/**
 * Plugin registry API.
 */
export interface PluginRegistry {
    /** Get all discovered plugins */
    getDiscovered(): readonly DiscoveredPlugin[];
    /** Get all loaded plugins */
    getLoaded(): readonly LoadedPlugin[];
    /** Get a specific plugin by name */
    get(name: string): LoadedPlugin | undefined;
    /** Check if a plugin is loaded */
    isLoaded(name: string): boolean;
    /** Load a plugin by name */
    load(name: string): Promise<LoadedPlugin>;
    /** Unload a plugin */
    unload(name: string): Promise<void>;
    /** Enable a plugin */
    enable(name: string): Promise<void>;
    /** Disable a plugin */
    disable(name: string): Promise<void>;
    /** Refresh plugin discovery */
    refresh(): Promise<readonly DiscoveredPlugin[]>;
}
//# sourceMappingURL=plugin.d.ts.map