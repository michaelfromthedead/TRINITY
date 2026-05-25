# FlowForge Testing Strategy

**Visual Programming Interface for Trinity Python Metaprogramming**

---

## Testing Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TESTING PYRAMID                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ▲                                                 │
│                          ╱ ╲                                                │
│                         ╱   ╲         E2E Tests (few, slow, high value)    │
│                        ╱ E2E ╲        - Full app flows                      │
│                       ╱───────╲       - Cross-platform                      │
│                      ╱         ╲                                            │
│                     ╱Integration╲     Integration Tests (some)              │
│                    ╱─────────────╲    - Tauri ↔ Python communication       │
│                   ╱               ╲   - IPC round trips                     │
│                  ╱                 ╲                                        │
│                 ╱    Unit Tests     ╲  Unit Tests (many, fast)             │
│                ╱─────────────────────╲ - Python: AST, codegen              │
│               ╱                       ╲- Rust: IPC, commands               │
│              ╱─────────────────────────╲- TS: Bridge, stores               │
│                                                                             │
│  SPECIAL TESTS:                                                             │
│  • Round-trip tests (parse → modify → generate → parse)                    │
│  • Trinity integration tests (decorator introspection)                      │
│  • Visual regression tests (canvas rendering)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Test Stack

| Layer | Framework | Location | Run Command |
|-------|-----------|----------|-------------|
| **Unit (Python)** | pytest | `flowforge_backend/tests/` | `pytest` |
| **Unit (Rust)** | cargo test | `src-tauri/src/` | `cargo test` |
| **Unit (TypeScript)** | Vitest | `apps/desktop/src/__tests__/` | `bun test` |
| **Integration** | pytest + Tauri | `tests/integration/` | `bun run test:integration` |
| **E2E** | Playwright + Tauri | `tests/e2e/` | `bun run test:e2e` |
| **Round-trip** | pytest | `flowforge_backend/tests/roundtrip/` | `pytest tests/roundtrip/` |

---

## 1. Python Unit Tests (pytest)

### 1.1 AST Parser Tests

**Location:** `flowforge_backend/tests/test_ast_parser.py`

```python
# flowforge_backend/tests/test_ast_parser.py

import pytest
from flowforge_backend.ast_parser import TrinityASTVisitor, parse_python_file

class TestTrinityASTVisitor:
    """Test extraction of Trinity patterns from Python AST."""

    def test_detects_component_decorator(self):
        source = '''
from trinity.decorators import component

@component
class Player:
    health: int = 100
    position: Vec3
'''
        visitor = TrinityASTVisitor()
        result = visitor.parse(source)

        assert len(result['nodes']) == 1
        assert result['nodes'][0]['type'] == 'Trinity/Component'
        assert result['nodes'][0]['title'] == 'Player'

    def test_extracts_component_fields(self):
        source = '''
@component
class Player:
    health: int = 100
    mana: int = 50
    name: str = "Hero"
'''
        visitor = TrinityASTVisitor()
        result = visitor.parse(source)

        fields = result['nodes'][0]['data']['fields']
        assert len(fields) == 3
        assert fields[0] == {'name': 'health', 'type': 'int', 'default': '100'}
        assert fields[1] == {'name': 'mana', 'type': 'int', 'default': '50'}
        assert fields[2] == {'name': 'name', 'type': 'str', 'default': '"Hero"'}

    def test_detects_system_decorator(self):
        source = '''
@system
class MovementSystem:
    def update(self, query: Query[Player, Velocity]):
        pass
'''
        visitor = TrinityASTVisitor()
        result = visitor.parse(source)

        assert len(result['nodes']) == 1
        assert result['nodes'][0]['type'] == 'Trinity/System'
        assert result['nodes'][0]['title'] == 'MovementSystem'

    def test_detects_resource_decorator(self):
        source = '''
@resource
class GameTime:
    delta: float = 0.0
    total: float = 0.0
'''
        visitor = TrinityASTVisitor()
        result = visitor.parse(source)

        assert result['nodes'][0]['type'] == 'Trinity/Resource'

    def test_detects_event_decorator(self):
        source = '''
@event
class PlayerDied:
    player_id: int
    killer_id: int
'''
        visitor = TrinityASTVisitor()
        result = visitor.parse(source)

        assert result['nodes'][0]['type'] == 'Trinity/Event'

    def test_handles_multiple_classes(self):
        source = '''
@component
class Player:
    health: int = 100

@component
class Enemy:
    damage: int = 10

@system
class CombatSystem:
    pass
'''
        visitor = TrinityASTVisitor()
        result = visitor.parse(source)

        assert len(result['nodes']) == 3

    def test_builds_edges_from_references(self):
        source = '''
@component
class Position:
    x: float = 0.0
    y: float = 0.0

@component
class Player:
    pos: Position
'''
        visitor = TrinityASTVisitor()
        result = visitor.parse(source)

        # Should have edge from Player.pos to Position
        edges = result['edges']
        assert any(e['from_field'] == 'pos' for e in edges)

    def test_handles_syntax_error_gracefully(self):
        source = '''
this is not valid python {{{
'''
        visitor = TrinityASTVisitor()

        with pytest.raises(SyntaxError):
            visitor.parse(source)

    def test_ignores_non_trinity_classes(self):
        source = '''
class RegularClass:
    def __init__(self):
        pass

@component
class TrinityClass:
    field: int = 0
'''
        visitor = TrinityASTVisitor()
        result = visitor.parse(source)

        # Only the Trinity class should be extracted
        assert len(result['nodes']) == 1
        assert result['nodes'][0]['title'] == 'TrinityClass'


class TestParseFile:
    """Test file-level parsing."""

    def test_parses_file(self, tmp_path):
        file = tmp_path / "test.py"
        file.write_text('''
@component
class TestComponent:
    value: int = 42
''')
        result = parse_python_file(str(file))

        assert result['source_file'] == str(file)
        assert len(result['nodes']) == 1

    def test_handles_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_python_file('/nonexistent/file.py')

    def test_handles_encoding(self, tmp_path):
        file = tmp_path / "unicode.py"
        file.write_text('''
@component
class Héros:  # Unicode in class name
    nom: str = "Éclair"
''', encoding='utf-8')

        result = parse_python_file(str(file))
        assert result['nodes'][0]['title'] == 'Héros'
```

### 1.2 Code Generation Tests

**Location:** `flowforge_backend/tests/test_codegen.py`

```python
# flowforge_backend/tests/test_codegen.py

import pytest
import ast
from flowforge_backend.codegen import generate_python, graph_to_ast

class TestGraphToAST:
    """Test conversion of node graph to Python AST."""

    def test_generates_component_class(self):
        graph = {
            'nodes': [{
                'id': 1,
                'type': 'Trinity/Component',
                'data': {
                    'class_name': 'Player',
                    'fields': [
                        {'name': 'health', 'type': 'int', 'default': '100'},
                        {'name': 'mana', 'type': 'int', 'default': '50'},
                    ]
                }
            }],
            'edges': []
        }

        code = generate_python(graph)

        assert '@component' in code
        assert 'class Player:' in code
        assert 'health: int = 100' in code
        assert 'mana: int = 50' in code

    def test_generates_system_class(self):
        graph = {
            'nodes': [{
                'id': 1,
                'type': 'Trinity/System',
                'data': {
                    'class_name': 'MovementSystem',
                    'methods': [{
                        'name': 'update',
                        'params': [{'name': 'query', 'type': 'Query[Player]'}],
                        'body': 'pass'
                    }]
                }
            }],
            'edges': []
        }

        code = generate_python(graph)

        assert '@system' in code
        assert 'class MovementSystem:' in code

    def test_generates_valid_python(self):
        graph = {
            'nodes': [{
                'id': 1,
                'type': 'Trinity/Component',
                'data': {
                    'class_name': 'Test',
                    'fields': [{'name': 'value', 'type': 'int', 'default': '0'}]
                }
            }],
            'edges': []
        }

        code = generate_python(graph)

        # Should parse without error
        ast.parse(code)

    def test_generates_imports(self):
        graph = {
            'nodes': [{
                'id': 1,
                'type': 'Trinity/Component',
                'data': {'class_name': 'Test', 'fields': []}
            }],
            'edges': []
        }

        code = generate_python(graph)

        assert 'from trinity.decorators import component' in code

    def test_handles_empty_graph(self):
        graph = {'nodes': [], 'edges': []}

        code = generate_python(graph)

        # Should still be valid Python (just imports maybe)
        ast.parse(code)

    def test_handles_multiple_nodes(self):
        graph = {
            'nodes': [
                {'id': 1, 'type': 'Trinity/Component', 'data': {'class_name': 'A', 'fields': []}},
                {'id': 2, 'type': 'Trinity/Component', 'data': {'class_name': 'B', 'fields': []}},
            ],
            'edges': []
        }

        code = generate_python(graph)

        assert 'class A:' in code
        assert 'class B:' in code


class TestCodegenEdgeCases:
    """Test edge cases in code generation."""

    def test_escapes_special_strings(self):
        graph = {
            'nodes': [{
                'id': 1,
                'type': 'Trinity/Component',
                'data': {
                    'class_name': 'Test',
                    'fields': [{'name': 's', 'type': 'str', 'default': '"hello\\nworld"'}]
                }
            }],
            'edges': []
        }

        code = generate_python(graph)
        ast.parse(code)  # Should not raise

    def test_handles_complex_types(self):
        graph = {
            'nodes': [{
                'id': 1,
                'type': 'Trinity/Component',
                'data': {
                    'class_name': 'Test',
                    'fields': [
                        {'name': 'items', 'type': 'List[int]', 'default': '[]'},
                        {'name': 'mapping', 'type': 'Dict[str, int]', 'default': '{}'},
                    ]
                }
            }],
            'edges': []
        }

        code = generate_python(graph)

        assert 'List[int]' in code
        assert 'Dict[str, int]' in code
```

### 1.3 Round-Trip Tests

**Location:** `flowforge_backend/tests/test_roundtrip.py`

```python
# flowforge_backend/tests/test_roundtrip.py

import pytest
import ast
from flowforge_backend.ast_parser import parse_python_source
from flowforge_backend.codegen import generate_python

class TestRoundTrip:
    """Test that parse → generate → parse produces equivalent results."""

    def test_simple_component_roundtrip(self):
        original = '''
from trinity.decorators import component

@component
class Player:
    health: int = 100
    mana: int = 50
'''
        # Parse to graph
        graph = parse_python_source(original)

        # Generate back to Python
        generated = generate_python(graph)

        # Parse the generated code
        graph2 = parse_python_source(generated)

        # Should have same structure
        assert len(graph['nodes']) == len(graph2['nodes'])
        assert graph['nodes'][0]['data']['fields'] == graph2['nodes'][0]['data']['fields']

    def test_multiple_classes_roundtrip(self):
        original = '''
@component
class Position:
    x: float = 0.0
    y: float = 0.0

@component
class Velocity:
    dx: float = 0.0
    dy: float = 0.0

@system
class MovementSystem:
    pass
'''
        graph = parse_python_source(original)
        generated = generate_python(graph)
        graph2 = parse_python_source(generated)

        assert len(graph['nodes']) == len(graph2['nodes'])

    def test_field_modifications_roundtrip(self):
        original = '''
@component
class Player:
    health: int = 100
'''
        # Parse
        graph = parse_python_source(original)

        # Modify - add a field
        graph['nodes'][0]['data']['fields'].append({
            'name': 'mana',
            'type': 'int',
            'default': '50'
        })

        # Generate
        generated = generate_python(graph)

        # Verify modification persisted
        assert 'mana: int = 50' in generated

        # Round-trip again
        graph2 = parse_python_source(generated)
        assert len(graph2['nodes'][0]['data']['fields']) == 2

    def test_preserves_field_order(self):
        original = '''
@component
class Test:
    a: int = 1
    b: int = 2
    c: int = 3
'''
        graph = parse_python_source(original)
        generated = generate_python(graph)
        graph2 = parse_python_source(generated)

        fields = graph2['nodes'][0]['data']['fields']
        assert [f['name'] for f in fields] == ['a', 'b', 'c']

    def test_data_loss_detection(self):
        """Ensure we don't lose information in round-trip."""
        original = '''
@component
class Player:
    health: int = 100
    name: str = "Hero"
    active: bool = True
'''
        graph = parse_python_source(original)
        generated = generate_python(graph)
        graph2 = parse_python_source(generated)

        # All fields should be preserved
        original_fields = graph['nodes'][0]['data']['fields']
        roundtrip_fields = graph2['nodes'][0]['data']['fields']

        assert original_fields == roundtrip_fields
```

### 1.4 Trinity Adapter Tests

**Location:** `flowforge_backend/tests/test_trinity_adapter.py`

```python
# flowforge_backend/tests/test_trinity_adapter.py

import pytest
from flowforge_backend.trinity_adapter import TrinityAdapter

# These tests require Trinity to be importable
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("trinity", reason="Trinity not installed"),
    reason="Trinity not installed"
)

class TestTrinityAdapter:
    """Test introspection of live Trinity runtime."""

    def test_discovers_registered_components(self):
        adapter = TrinityAdapter()
        components = adapter.get_components()

        # Should find at least one component from Trinity
        assert isinstance(components, list)

    def test_exports_component_schema(self):
        adapter = TrinityAdapter()
        schema = adapter.export_schema()

        assert isinstance(schema, dict)
        # Schema should have ComfyUI-compatible format
        for node_type, node_def in schema.items():
            assert 'input' in node_def
            assert 'output' in node_def

    def test_introspects_component_fields(self):
        # Create a test component
        from trinity.decorators import component

        @component
        class TestPlayer:
            health: int = 100

        adapter = TrinityAdapter()
        fields = adapter.get_component_fields(TestPlayer)

        assert len(fields) == 1
        assert fields[0]['name'] == 'health'
        assert fields[0]['type'] == 'int'
```

### 1.5 IPC Protocol Tests

**Location:** `flowforge_backend/tests/test_ipc.py`

```python
# flowforge_backend/tests/test_ipc.py

import pytest
import json
from flowforge_backend.ipc import IPCHandler, IPCMessage

class TestIPCMessage:
    """Test IPC message serialization."""

    def test_request_serialization(self):
        msg = IPCMessage.request("123", "get_object_info", {})
        json_str = msg.to_json()

        parsed = json.loads(json_str)
        assert parsed['id'] == "123"
        assert parsed['type'] == "request"
        assert parsed['method'] == "get_object_info"

    def test_response_serialization(self):
        msg = IPCMessage.response("123", {"nodes": {}})
        json_str = msg.to_json()

        parsed = json.loads(json_str)
        assert parsed['type'] == "response"
        assert 'result' in parsed

    def test_error_serialization(self):
        msg = IPCMessage.error("123", -32600, "Invalid Request")
        json_str = msg.to_json()

        parsed = json.loads(json_str)
        assert 'error' in parsed
        assert parsed['error']['code'] == -32600


class TestIPCHandler:
    """Test IPC request handling."""

    def test_handles_ping(self):
        handler = IPCHandler()
        request = '{"id": "1", "type": "request", "method": "ping"}'

        response = handler.handle(request)
        parsed = json.loads(response)

        assert parsed['result'] == 'pong'

    def test_handles_unknown_method(self):
        handler = IPCHandler()
        request = '{"id": "1", "type": "request", "method": "nonexistent"}'

        response = handler.handle(request)
        parsed = json.loads(response)

        assert 'error' in parsed
        assert parsed['error']['code'] == -32601  # Method not found

    def test_handles_parse_python_file(self, tmp_path):
        file = tmp_path / "test.py"
        file.write_text('@component\nclass Test:\n    x: int = 0')

        handler = IPCHandler()
        request = json.dumps({
            "id": "1",
            "type": "request",
            "method": "parse_python_file",
            "params": {"path": str(file)}
        })

        response = handler.handle(request)
        parsed = json.loads(response)

        assert 'result' in parsed
        assert 'nodes' in parsed['result']
```

---

## 2. Rust Unit Tests

### 2.1 IPC Protocol Tests

**Location:** `src-tauri/src/sidecar/protocol.rs`

```rust
// src-tauri/src/sidecar/protocol.rs

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize)]
pub struct IPCMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<IPCError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IPCError {
    pub code: i32,
    pub message: String,
}

impl IPCMessage {
    pub fn request(id: &str, method: &str, params: Value) -> Self {
        Self {
            id: id.to_string(),
            msg_type: "request".to_string(),
            method: Some(method.to_string()),
            params: Some(params),
            result: None,
            error: None,
        }
    }

    pub fn response(id: &str, result: Value) -> Self {
        Self {
            id: id.to_string(),
            msg_type: "response".to_string(),
            method: None,
            params: None,
            result: Some(result),
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_request_serialization() {
        let msg = IPCMessage::request("123", "parse_python_file", json!({"path": "/test.py"}));
        let json = serde_json::to_string(&msg).unwrap();

        assert!(json.contains("\"id\":\"123\""));
        assert!(json.contains("\"type\":\"request\""));
        assert!(json.contains("\"method\":\"parse_python_file\""));
    }

    #[test]
    fn test_response_serialization() {
        let msg = IPCMessage::response("123", json!({"nodes": []}));
        let json = serde_json::to_string(&msg).unwrap();

        assert!(json.contains("\"type\":\"response\""));
        assert!(json.contains("\"result\""));
    }

    #[test]
    fn test_roundtrip() {
        let original = IPCMessage::request("abc", "test", json!({"key": "value"}));
        let json = serde_json::to_string(&original).unwrap();
        let parsed: IPCMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(original.id, parsed.id);
        assert_eq!(original.method, parsed.method);
    }
}
```

### 2.2 Python Sidecar Manager Tests

```rust
// src-tauri/src/sidecar/mod.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_sidecar_spawn() {
        // This test requires Python to be installed
        let result = PythonSidecar::spawn();

        match result {
            Ok(mut sidecar) => {
                // Send ping
                let response = sidecar.send_request("ping", json!({}));
                assert!(response.is_ok());
                sidecar.shutdown();
            }
            Err(e) => {
                // Python not available, skip
                eprintln!("Skipping test: {}", e);
            }
        }
    }

    #[test]
    fn test_sidecar_timeout() {
        // Test that requests timeout appropriately
        // Implementation details...
    }
}
```

---

## 3. Integration Tests

### 3.1 Tauri ↔ Python IPC Tests

**Location:** `tests/integration/test_ipc.py`

```python
# tests/integration/test_ipc.py

import pytest
import subprocess
import json
import time

class TestTauriPythonIPC:
    """Test full IPC round-trip between Tauri and Python."""

    @pytest.fixture
    def python_sidecar(self):
        """Start Python sidecar for testing."""
        proc = subprocess.Popen(
            ['python', '-m', 'flowforge_backend'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(0.5)  # Wait for startup
        yield proc
        proc.terminate()

    def test_ping_pong(self, python_sidecar):
        request = json.dumps({
            "id": "1",
            "type": "request",
            "method": "ping"
        }) + "\n"

        python_sidecar.stdin.write(request)
        python_sidecar.stdin.flush()

        response = python_sidecar.stdout.readline()
        parsed = json.loads(response)

        assert parsed['result'] == 'pong'

    def test_parse_file(self, python_sidecar, tmp_path):
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text('''
@component
class Player:
    health: int = 100
''')

        request = json.dumps({
            "id": "2",
            "type": "request",
            "method": "parse_python_file",
            "params": {"path": str(test_file)}
        }) + "\n"

        python_sidecar.stdin.write(request)
        python_sidecar.stdin.flush()

        response = python_sidecar.stdout.readline()
        parsed = json.loads(response)

        assert 'result' in parsed
        assert len(parsed['result']['nodes']) == 1

    def test_generate_python(self, python_sidecar):
        graph = {
            "nodes": [{
                "id": 1,
                "type": "Trinity/Component",
                "data": {
                    "class_name": "Test",
                    "fields": [{"name": "x", "type": "int", "default": "0"}]
                }
            }],
            "edges": []
        }

        request = json.dumps({
            "id": "3",
            "type": "request",
            "method": "generate_python",
            "params": {"graph": graph}
        }) + "\n"

        python_sidecar.stdin.write(request)
        python_sidecar.stdin.flush()

        response = python_sidecar.stdout.readline()
        parsed = json.loads(response)

        assert 'result' in parsed
        assert 'class Test:' in parsed['result']['code']
```

---

## 4. End-to-End Tests

### 4.1 Playwright + Tauri Setup

**Configuration:** `playwright.config.ts`

```typescript
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30000,
  expect: { timeout: 5000 },
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: 'html',
  use: {
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
});
```

### 4.2 E2E Test Cases

```typescript
// tests/e2e/open-python.spec.ts

import { test, expect } from '@playwright/test';
import { writeFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

test.describe('Open Python File', () => {
  test('can open Python file and see nodes', async ({ page }) => {
    // Create test Python file
    const testFile = join(tmpdir(), 'test-component.py');
    writeFileSync(testFile, `
@component
class Player:
    health: int = 100
    mana: int = 50
`);

    await page.goto('tauri://localhost');
    await page.waitForSelector('canvas');

    // Open file via menu or drag-drop
    // ... implementation details

    // Verify node appears on canvas
    const node = page.locator('[data-node-type="Trinity/Component"]');
    await expect(node).toBeVisible();
  });

  test('can edit node and regenerate Python', async ({ page }) => {
    await page.goto('tauri://localhost');
    await page.waitForSelector('canvas');

    // Open file, edit node, save
    // ... implementation details

    // Verify generated Python is valid
  });
});
```

---

## 5. Visual Regression Tests

```typescript
// tests/visual/canvas.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Canvas Visual Regression', () => {
  test('empty canvas matches snapshot', async ({ page }) => {
    await page.goto('tauri://localhost');
    await page.waitForSelector('canvas');
    await page.waitForTimeout(500);

    const canvas = page.locator('canvas');
    await expect(canvas).toHaveScreenshot('empty-canvas.png');
  });

  test('Trinity component node matches snapshot', async ({ page }) => {
    await page.goto('tauri://localhost');
    await page.waitForSelector('canvas');

    // Add a Trinity component node
    // ... implementation

    await page.waitForTimeout(500);
    const canvas = page.locator('canvas');
    await expect(canvas).toHaveScreenshot('component-node.png', {
      maxDiffPixels: 100,
    });
  });
});
```

---

## 6. Manual QA Checklist

```markdown
## FlowForge v{VERSION} QA Checklist

### Installation
- [ ] Windows: MSI installs without errors
- [ ] macOS: DMG mounts and installs correctly
- [ ] Linux: AppImage runs on Ubuntu 22.04
- [ ] Python sidecar starts automatically

### File Operations
- [ ] Open .py file via File menu
- [ ] Open .py file via drag-and-drop
- [ ] Save workflow to file
- [ ] Recent files list works

### AST Parsing
- [ ] @component classes become nodes
- [ ] @system classes become nodes
- [ ] @resource classes become nodes
- [ ] @event classes become nodes
- [ ] Field types displayed correctly
- [ ] Default values shown

### Code Generation
- [ ] New field → regenerates Python
- [ ] Remove field → regenerates Python
- [ ] Rename class → regenerates Python
- [ ] Generated code is valid Python
- [ ] Diff preview shows changes

### Canvas
- [ ] Nodes render correctly
- [ ] Pan and zoom work
- [ ] Connections between nodes visible
- [ ] Node selection works

### Sign-off
- Tester: _______________
- Date: _______________
- Pass/Fail: _______________
```

---

## 7. CI Configuration

```yaml
# .github/workflows/test.yml

name: Test

on:
  push:
    branches: [main]
  pull_request:

jobs:
  python-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ./flowforge_backend[dev]
      - run: pytest flowforge_backend/tests/ -v

  rust-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rust-lang/setup-rust-toolchain@v1
      - run: cargo test
        working-directory: apps/desktop/src-tauri

  frontend-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: oven-sh/setup-bun@v1
      - run: bun install
      - run: bun test

  integration-test:
    runs-on: ubuntu-latest
    needs: [python-test, rust-test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - uses: oven-sh/setup-bun@v1
      - run: pip install -e ./flowforge_backend
      - run: bun install
      - run: bun run test:integration

  e2e-test:
    runs-on: ${{ matrix.os }}
    needs: [integration-test]
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - uses: oven-sh/setup-bun@v1
      - run: bunx playwright install --with-deps
      - run: bun run test:e2e
```

---

## 8. Test Commands Summary

```bash
# Python tests
cd flowforge_backend
pytest                              # All Python tests
pytest tests/test_ast_parser.py    # AST parser tests
pytest tests/test_codegen.py       # Code generation tests
pytest tests/test_roundtrip.py     # Round-trip tests
pytest -v --cov                    # With coverage

# Rust tests
cd apps/desktop/src-tauri
cargo test                         # All Rust tests

# Frontend tests
bun test                           # TypeScript tests

# Integration tests
bun run test:integration           # Tauri ↔ Python IPC

# E2E tests
bun run test:e2e                   # Full app tests

# Visual regression
bun run test:visual                # Screenshot comparisons

# All tests
bun run test                       # Run everything
```

---

## Summary

| Test Type | Framework | Target Count | Purpose |
|-----------|-----------|--------------|---------|
| Python Unit | pytest | 50+ | AST parsing, codegen, IPC |
| Rust Unit | cargo test | 20+ | Sidecar manager, protocol |
| TS Unit | Vitest | 30+ | Bridge, stores |
| Integration | pytest | 20+ | Tauri ↔ Python IPC |
| Round-trip | pytest | 10+ | Parse → modify → generate |
| E2E | Playwright | 15+ | Full app workflows |
| Visual | Playwright | 10+ | Canvas rendering |

**Total CI time target:** < 15 minutes
