/**
 * Integration Test: Python Sidecar IPC Round-Trip
 *
 * This script verifies the full round-trip communication between the TypeScript
 * frontend and Python backend via JSON-RPC 2.0 protocol:
 *
 *   Tauri -> Python IPC -> parse/generate -> return
 *
 * Tests:
 *   1. Spawns the Python sidecar manually
 *   2. Sends a parse_python_file request with a test fixture
 *   3. Verifies the response contains valid node graph
 *   4. Sends a generate_code request with the graph
 *   5. Verifies the response contains valid Python code
 *   6. Sends validate_code request and verifies success
 *   7. Tests error handling for invalid inputs
 *
 * Note: Timeout values mirror TEST_CONFIG in src/config/flowforge.config.ts
 * Keep them in sync when updating.
 *
 * Usage:
 *   npx ts-node scripts/integration-test.ts
 *   # or
 *   npx tsx scripts/integration-test.ts
 *
 * @module integration-test
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// =============================================================================
// Test Configuration
// =============================================================================

// Test configuration - mirrors values from @/config/flowforge.config.ts
const TEST_TIMEOUTS = {
  /** Timeout for IPC responses (ms) */
  ipcTimeout: 10000,
  /** Timeout for sidecar startup (ms) */
  sidecarStartup: 5000,
} as const;

// =============================================================================
// Types
// =============================================================================

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number | string;
  method: string;
  params: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number | string;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

interface NodeGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata?: Record<string, unknown>;
}

interface GraphNode {
  id: string;
  type: "component" | "system" | "resource" | "event";
  name: string;
  position: [number, number];
  data: Record<string, unknown>;
  source?: { file: string; line: number };
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: "reference" | "inheritance" | "query";
}

interface ValidationResult {
  success: boolean;
  errors: Array<{ message: string; line?: number; column?: number }>;
  warnings: Array<{ message: string; line?: number; column?: number }>;
  source_hash?: string;
}

interface ParseResult {
  success: boolean;
  errors: string[];
  graph: NodeGraph | null;
}

interface GenerateResult {
  source: string;
  validation: ValidationResult;
  node_count: number;
}

interface TestResult {
  name: string;
  passed: boolean;
  duration: number;
  error?: string;
}

// =============================================================================
// Constants
// =============================================================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Paths relative to the script location
const PROJECT_ROOT = join(__dirname, "..", "..", "..");
const BACKEND_MODULE = join(PROJECT_ROOT, "flowforge_backend");
const FIXTURES_DIR = join(__dirname, "..", "test-fixtures");

// Test fixture file path - uses existing fixture in test-fixtures directory
const TEST_FIXTURE_FILE = join(FIXTURES_DIR, "test_trinity_ecs.py");

// Sample graph for code generation test
const SAMPLE_GRAPH: NodeGraph = {
  nodes: [
    {
      id: "node_1",
      type: "component",
      name: "TestComponent",
      position: [100, 100],
      data: {
        fields: [
          { name: "value", type_annotation: "int", default_value: "42" },
          { name: "name", type_annotation: "str", default_value: '"test"' },
        ],
        docstring: "A test component for integration testing.",
      },
    },
    {
      id: "node_2",
      type: "system",
      name: "TestSystem",
      position: [300, 100],
      data: {
        methods: [
          {
            name: "process",
            parameters: [],
            return_type: "None",
            query_types: ["TestComponent"],
            docstring: "Process test components.",
          },
        ],
        docstring: "A test system for integration testing.",
      },
    },
    {
      id: "node_3",
      type: "resource",
      name: "TestResource",
      position: [100, 300],
      data: {
        fields: [{ name: "counter", type_annotation: "int", default_value: "0" }],
        is_singleton: true,
      },
    },
    {
      id: "node_4",
      type: "event",
      name: "TestEvent",
      position: [300, 300],
      data: {
        payload_fields: [
          { name: "event_id", type_annotation: "str" },
          { name: "timestamp", type_annotation: "float" },
        ],
      },
    },
  ],
  edges: [
    {
      id: "edge_1",
      source: "node_2",
      target: "node_1",
      type: "query",
    },
  ],
};

// =============================================================================
// Python Sidecar Manager
// =============================================================================

class PythonSidecar {
  private process: ChildProcess | null = null;
  private readline: Interface | null = null;
  private requestId = 0;
  private pendingRequests = new Map<
    number | string,
    { resolve: (value: JsonRpcResponse) => void; reject: (error: Error) => void }
  >();
  private ready = false;

  constructor(
    private pythonPath: string = "python3",
    private backendModule: string = BACKEND_MODULE
  ) {}

  /**
   * Start the Python sidecar process.
   */
  async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Spawn Python module as subprocess
      this.process = spawn(this.pythonPath, ["-m", "flowforge_backend"], {
        cwd: PROJECT_ROOT,
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          FLOWFORGE_DEBUG: "1",
        },
      });

      if (!this.process.stdout || !this.process.stdin) {
        reject(new Error("Failed to create stdio pipes"));
        return;
      }

      // Setup readline for line-delimited JSON responses
      this.readline = createInterface({
        input: this.process.stdout,
        crlfDelay: Infinity,
      });

      // Handle incoming responses
      this.readline.on("line", (line) => {
        this.handleResponse(line);
      });

      // Handle stderr (logging)
      this.process.stderr?.on("data", (data) => {
        const message = data.toString().trim();
        if (message.includes("Ready for requests")) {
          this.ready = true;
          resolve();
        }
        // Optionally log backend messages
        if (process.env.DEBUG) {
          console.log(`[BACKEND] ${message}`);
        }
      });

      // Handle process errors
      this.process.on("error", (error) => {
        reject(new Error(`Failed to start Python sidecar: ${error.message}`));
      });

      this.process.on("close", (code) => {
        if (code !== 0 && code !== null) {
          console.error(`Python sidecar exited with code ${code}`);
        }
        this.ready = false;
      });

      // Timeout if not ready
      setTimeout(() => {
        if (!this.ready) {
          this.stop();
          reject(new Error("Python sidecar startup timeout"));
        }
      }, TEST_TIMEOUTS.sidecarStartup);
    });
  }

  /**
   * Stop the Python sidecar process.
   */
  stop(): void {
    if (this.process) {
      this.process.stdin?.end();
      this.process.kill();
      this.process = null;
    }
    if (this.readline) {
      this.readline.close();
      this.readline = null;
    }
    this.ready = false;
  }

  /**
   * Send a JSON-RPC request and wait for response.
   */
  async request<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    if (!this.process?.stdin || !this.ready) {
      throw new Error("Python sidecar not running");
    }

    const id = ++this.requestId;
    const request: JsonRpcRequest = {
      jsonrpc: "2.0",
      id,
      method,
      params,
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, {
        resolve: (response) => {
          if (response.error) {
            reject(new Error(`RPC Error ${response.error.code}: ${response.error.message}`));
          } else {
            resolve(response.result as T);
          }
        },
        reject,
      });

      // Send request as line-delimited JSON
      const requestLine = JSON.stringify(request) + "\n";
      this.process!.stdin!.write(requestLine);
    });
  }

  /**
   * Handle incoming JSON-RPC response.
   */
  private handleResponse(line: string): void {
    try {
      const response: JsonRpcResponse = JSON.parse(line);
      const pending = this.pendingRequests.get(response.id);

      if (pending) {
        this.pendingRequests.delete(response.id);
        pending.resolve(response);
      }
    } catch (error) {
      console.error(`Failed to parse response: ${line}`);
    }
  }

  /**
   * Check if the sidecar is running and ready.
   */
  isReady(): boolean {
    return this.ready;
  }
}

// =============================================================================
// Test Runner
// =============================================================================

class IntegrationTestRunner {
  private sidecar: PythonSidecar;
  private results: TestResult[] = [];
  private fixtureFile: string | null = null;

  constructor() {
    this.sidecar = new PythonSidecar();
  }

  /**
   * Setup test fixtures.
   * Uses existing fixture file from test-fixtures directory.
   */
  private setupFixtures(): void {
    this.fixtureFile = TEST_FIXTURE_FILE;

    if (!existsSync(this.fixtureFile)) {
      throw new Error(
        `Test fixture not found: ${this.fixtureFile}\n` +
        `Please ensure test-fixtures/test_trinity_ecs.py exists.`
      );
    }

    console.log(`Using test fixture: ${this.fixtureFile}`);
  }

  /**
   * Cleanup test fixtures.
   * No cleanup needed since we use the existing fixture file.
   */
  private cleanupFixtures(): void {
    // No cleanup needed - we use the existing fixture file
    console.log(`Test fixture preserved: ${this.fixtureFile}`);
  }

  /**
   * Run a single test case.
   */
  private async runTest(name: string, fn: () => Promise<void>): Promise<void> {
    const start = Date.now();
    try {
      await fn();
      this.results.push({
        name,
        passed: true,
        duration: Date.now() - start,
      });
      console.log(`  [PASS] ${name} (${Date.now() - start}ms)`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.results.push({
        name,
        passed: false,
        duration: Date.now() - start,
        error: errorMessage,
      });
      console.log(`  [FAIL] ${name} (${Date.now() - start}ms)`);
      console.log(`         Error: ${errorMessage}`);
    }
  }

  /**
   * Assert helper.
   */
  private assert(condition: boolean, message: string): void {
    if (!condition) {
      throw new Error(`Assertion failed: ${message}`);
    }
  }

  /**
   * Test 1: Ping - Basic connectivity test.
   */
  private async testPing(): Promise<void> {
    const result = await this.sidecar.request<{ pong: boolean; timestamp: number }>("ping");

    this.assert(result.pong === true, "Expected pong to be true");
    this.assert(typeof result.timestamp === "number", "Expected timestamp to be a number");
  }

  /**
   * Test 2: Get Version - Protocol version verification.
   */
  private async testGetVersion(): Promise<void> {
    const result = await this.sidecar.request<{
      version: string;
      python_version: string;
      protocol_version: string;
    }>("get_version");

    this.assert(typeof result.version === "string", "Expected version to be a string");
    this.assert(typeof result.python_version === "string", "Expected python_version to be a string");
    this.assert(result.protocol_version === "1.0.0", "Expected protocol version 1.0.0");
  }

  /**
   * Test 3: List Methods - Verify available methods.
   */
  private async testListMethods(): Promise<void> {
    const result = await this.sidecar.request<{ methods: string[] }>("list_methods");

    this.assert(Array.isArray(result.methods), "Expected methods to be an array");
    this.assert(result.methods.includes("ping"), "Expected methods to include 'ping'");
    this.assert(result.methods.includes("parse_python_file"), "Expected methods to include 'parse_python_file'");
    this.assert(result.methods.includes("validate_python"), "Expected methods to include 'validate_python'");
    this.assert(result.methods.includes("generate_python"), "Expected methods to include 'generate_python'");
  }

  /**
   * Test 4: Parse Python File - Parse test fixture.
   */
  private async testParsePythonFile(): Promise<ParseResult> {
    this.assert(this.fixtureFile !== null, "Fixture file not set up");

    const result = await this.sidecar.request<ParseResult>("parse_python_file", {
      path: this.fixtureFile,
    });

    this.assert(result.success === true, `Parse failed: ${result.errors?.join(", ")}`);
    this.assert(result.graph !== null, "Expected graph to be non-null");
    this.assert(Array.isArray(result.graph!.nodes), "Expected graph.nodes to be an array");
    this.assert(result.graph!.nodes.length > 0, "Expected at least one node");

    // Verify node types
    const nodeTypes = new Set(result.graph!.nodes.map((n) => n.type));
    this.assert(nodeTypes.has("component"), "Expected at least one component node");
    this.assert(nodeTypes.has("system"), "Expected at least one system node");
    this.assert(nodeTypes.has("resource"), "Expected at least one resource node");
    this.assert(nodeTypes.has("event"), "Expected at least one event node");

    // Verify specific nodes
    const positionNode = result.graph!.nodes.find((n) => n.name === "Position");
    this.assert(positionNode !== undefined, "Expected to find Position component");
    this.assert(positionNode!.type === "component", "Position should be a component");

    const movementSystem = result.graph!.nodes.find((n) => n.name === "MovementSystem");
    this.assert(movementSystem !== undefined, "Expected to find MovementSystem");
    this.assert(movementSystem!.type === "system", "MovementSystem should be a system");

    return result;
  }

  /**
   * Test 5: Generate Code from Graph - Code generation.
   */
  private async testGenerateCode(): Promise<GenerateResult> {
    const result = await this.sidecar.request<GenerateResult>("generate_python", {
      graph: SAMPLE_GRAPH,
      format_with_black: false, // Avoid black dependency in test
      add_header: true,
    });

    this.assert(typeof result.source === "string", "Expected source to be a string");
    this.assert(result.source.length > 0, "Expected non-empty source code");

    // Verify generated code contains expected elements
    this.assert(result.source.includes("@component"), "Expected @component decorator");
    this.assert(result.source.includes("class TestComponent"), "Expected TestComponent class");
    this.assert(result.source.includes("@system"), "Expected @system decorator");
    this.assert(result.source.includes("class TestSystem"), "Expected TestSystem class");
    this.assert(result.source.includes("@resource"), "Expected @resource decorator");
    this.assert(result.source.includes("class TestResource"), "Expected TestResource class");
    this.assert(result.source.includes("@event"), "Expected @event decorator");
    this.assert(result.source.includes("class TestEvent"), "Expected TestEvent class");

    // Verify validation result
    this.assert(result.validation !== undefined, "Expected validation result");
    this.assert(result.validation.success === true, "Expected validation to succeed");

    // Verify node count
    this.assert(result.node_count === 4, `Expected 4 nodes, got ${result.node_count}`);

    return result;
  }

  /**
   * Test 6: Validate Code - Code validation.
   */
  private async testValidateCode(source: string): Promise<void> {
    const result = await this.sidecar.request<ValidationResult>("validate_python", {
      source,
      check_semantics: false,
    });

    this.assert(result.success === true, `Validation failed: ${result.errors?.map((e) => e.message).join(", ")}`);
    this.assert(Array.isArray(result.errors), "Expected errors to be an array");
    this.assert(result.errors.length === 0, "Expected no validation errors");
  }

  /**
   * Test 7: Validate Invalid Code - Error handling.
   */
  private async testValidateInvalidCode(): Promise<void> {
    const invalidSource = `
class Broken
    x: int = syntax error here!!!
`;

    const result = await this.sidecar.request<ValidationResult>("validate_python", {
      source: invalidSource,
    });

    this.assert(result.success === false, "Expected validation to fail for invalid code");
    this.assert(result.errors.length > 0, "Expected validation errors");
  }

  /**
   * Test 8: Parse Non-existent File - Error handling.
   */
  private async testParseNonExistentFile(): Promise<void> {
    const result = await this.sidecar.request<ParseResult>("parse_python_file", {
      path: "/non/existent/file.py",
    });

    this.assert(result.success === false, "Expected parse to fail for non-existent file");
    this.assert(result.errors.length > 0, "Expected error messages");
    this.assert(result.graph === null, "Expected graph to be null");
  }

  /**
   * Test 9: Generate Code with Empty Graph - Edge case.
   */
  private async testGenerateEmptyGraph(): Promise<void> {
    const emptyGraph: NodeGraph = {
      nodes: [],
      edges: [],
    };

    const result = await this.sidecar.request<GenerateResult>("generate_python", {
      graph: emptyGraph,
    });

    this.assert(typeof result.source === "string", "Expected source to be a string");
    this.assert(result.validation.success === true, "Expected validation to succeed for empty graph");
    this.assert(result.node_count === 0, "Expected 0 nodes");
  }

  /**
   * Test 10: Invalid Method - Error handling.
   */
  private async testInvalidMethod(): Promise<void> {
    try {
      await this.sidecar.request("non_existent_method");
      throw new Error("Expected error for non-existent method");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.assert(message.includes("-32601") || message.includes("Method not found"), "Expected method not found error");
    }
  }

  /**
   * Test 11: Invalid Params - Error handling.
   */
  private async testInvalidParams(): Promise<void> {
    try {
      await this.sidecar.request("parse_python_file", {});
      throw new Error("Expected error for missing params");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.assert(message.includes("-32602") || message.includes("path"), "Expected invalid params error");
    }
  }

  /**
   * Test 12: Round-Trip - Parse, Generate, Validate cycle.
   */
  private async testRoundTrip(): Promise<void> {
    // Step 1: Parse the fixture file
    const parseResult = await this.sidecar.request<ParseResult>("parse_python_file", {
      path: this.fixtureFile,
    });
    this.assert(parseResult.success, "Round-trip parse failed");

    // Step 2: Generate code from the parsed graph
    const generateResult = await this.sidecar.request<GenerateResult>("generate_python", {
      graph: parseResult.graph,
      format_with_black: false,
    });
    this.assert(generateResult.validation.success, "Round-trip generation failed");

    // Step 3: Validate the generated code
    const validateResult = await this.sidecar.request<ValidationResult>("validate_python", {
      source: generateResult.source,
    });
    this.assert(validateResult.success, "Round-trip validation failed");

    // Step 4: Verify key elements are preserved
    this.assert(generateResult.source.includes("class Position"), "Position class not preserved");
    this.assert(generateResult.source.includes("class Velocity"), "Velocity class not preserved");
    this.assert(generateResult.source.includes("class Health"), "Health class not preserved");
    this.assert(generateResult.source.includes("class GameConfig"), "GameConfig class not preserved");
    this.assert(generateResult.source.includes("class PlayerDied"), "PlayerDied class not preserved");
    this.assert(generateResult.source.includes("class MovementSystem"), "MovementSystem class not preserved");
  }

  /**
   * Test 13: Generate Diff - Diff generation.
   */
  private async testGenerateDiff(): Promise<void> {
    const original = `@component
class Position:
    x: float = 0.0
    y: float = 0.0
`;

    const modified = `@component
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
`;

    const result = await this.sidecar.request<{
      hasChanges: boolean;
      unifiedDiff: string;
      stats: { additions: number; deletions: number; changes: number };
      filename: string;
    }>("generate_diff", {
      original,
      modified,
      filename: "test.py",
    });

    this.assert(result.hasChanges === true, "Expected diff to have changes");
    this.assert(result.stats.additions > 0, "Expected lines added");
    this.assert(result.unifiedDiff.includes("+    z: float"), "Expected diff to show added line");
  }

  /**
   * Run all integration tests.
   */
  async run(): Promise<void> {
    console.log("\n========================================");
    console.log("FlowForge Integration Test Suite");
    console.log("========================================\n");

    try {
      // Setup
      console.log("Setting up test fixtures...");
      this.setupFixtures();

      console.log("Starting Python sidecar...");
      await this.sidecar.start();
      console.log("Python sidecar ready.\n");

      console.log("Running tests...\n");

      // Basic connectivity tests
      await this.runTest("1. Ping - Basic connectivity", () => this.testPing());
      await this.runTest("2. Get Version - Protocol verification", () => this.testGetVersion());
      await this.runTest("3. List Methods - Available methods", () => this.testListMethods());

      // Core functionality tests
      let parseResult: ParseResult | null = null;
      await this.runTest("4. Parse Python File - Parse test fixture", async () => {
        parseResult = await this.testParsePythonFile();
      });

      let generateResult: GenerateResult | null = null;
      await this.runTest("5. Generate Code - Code generation from graph", async () => {
        generateResult = await this.testGenerateCode();
      });

      await this.runTest("6. Validate Code - Validate generated code", async () => {
        if (generateResult) {
          await this.testValidateCode(generateResult.source);
        }
      });

      // Error handling tests
      await this.runTest("7. Validate Invalid Code - Error handling", () => this.testValidateInvalidCode());
      await this.runTest("8. Parse Non-existent File - Error handling", () => this.testParseNonExistentFile());
      await this.runTest("9. Generate Empty Graph - Edge case", () => this.testGenerateEmptyGraph());
      await this.runTest("10. Invalid Method - Error handling", () => this.testInvalidMethod());
      await this.runTest("11. Invalid Params - Error handling", () => this.testInvalidParams());

      // Integration tests
      await this.runTest("12. Round-Trip - Parse/Generate/Validate cycle", () => this.testRoundTrip());
      await this.runTest("13. Generate Diff - Diff generation", () => this.testGenerateDiff());

      // Print summary
      this.printSummary();
    } finally {
      // Cleanup
      console.log("\nCleaning up...");
      this.sidecar.stop();
      this.cleanupFixtures();
      console.log("Done.\n");
    }
  }

  /**
   * Print test summary.
   */
  private printSummary(): void {
    console.log("\n========================================");
    console.log("Test Summary");
    console.log("========================================\n");

    const passed = this.results.filter((r) => r.passed).length;
    const failed = this.results.filter((r) => !r.passed).length;
    const totalDuration = this.results.reduce((sum, r) => sum + r.duration, 0);

    console.log(`Total:    ${this.results.length} tests`);
    console.log(`Passed:   ${passed} tests`);
    console.log(`Failed:   ${failed} tests`);
    console.log(`Duration: ${totalDuration}ms`);
    console.log();

    if (failed > 0) {
      console.log("Failed tests:");
      for (const result of this.results.filter((r) => !r.passed)) {
        console.log(`  - ${result.name}`);
        console.log(`    Error: ${result.error}`);
      }
      console.log();
    }

    // Exit with appropriate code
    if (failed > 0) {
      process.exitCode = 1;
    }
  }
}

// =============================================================================
// Main Entry Point
// =============================================================================

async function main(): Promise<void> {
  const runner = new IntegrationTestRunner();
  await runner.run();
}

// Run if executed directly
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
