#!/usr/bin/env python3
"""Validate <library>_decisions.json against the schema in PHASE_02_CONTRACT.md#T-02.2.

Usage: python validate_decisions_schema.py <decisions_json_path>
Exit 0 = valid; exit 1 = schema violation.
"""

import json
import re
import sys
from pathlib import Path

UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")
REQUIRED_META = {"library", "domain", "version", "created"}
REQUIRED_PORT_TYPE = {"name", "description", "examples"}
REQUIRED_PHASE = {"name", "order", "description"}
REQUIRED_ATOM = {"name", "phase", "inputs", "outputs", "description"}


def fail(msg: str) -> None:
    print(f"INVALID: {msg}", file=sys.stderr)
    sys.exit(1)


def main(path: str) -> None:
    p = Path(path)
    if not p.exists():
        fail(f"file not found: {p}")
    if not p.name.endswith("_decisions.json"):
        fail(f"file must be named *_decisions.json, got {p.name!r}")

    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        fail(f"invalid JSON: {e}")

    # Meta
    meta = d.get("meta", {})
    missing = REQUIRED_META - set(meta.keys())
    if missing:
        fail(f"meta missing keys: {sorted(missing)}")

    # Port types
    port_types = d.get("port_types", [])
    if not isinstance(port_types, list):
        fail("port_types must be a list")
    if len(port_types) < 3:
        fail(f"port_types cardinality {len(port_types)} < 3 minimum")
    port_names = set()
    for i, pt in enumerate(port_types):
        missing = REQUIRED_PORT_TYPE - set(pt.keys())
        if missing:
            fail(f"port_types[{i}] missing keys: {sorted(missing)}")
        if not UPPER_SNAKE.match(pt["name"]):
            fail(f"port_types[{i}].name {pt['name']!r} not UPPER_SNAKE")
        if pt["name"] in port_names:
            fail(f"duplicate port_type name {pt['name']!r}")
        port_names.add(pt["name"])
        if not isinstance(pt["examples"], list):
            fail(f"port_types[{i}].examples must be a list")

    # Phases
    phases = d.get("phases", [])
    if not isinstance(phases, list):
        fail("phases must be a list")
    if len(phases) < 3:
        fail(f"phases cardinality {len(phases)} < 3 minimum")
    phase_names = set()
    phase_orders = []
    for i, ph in enumerate(phases):
        missing = REQUIRED_PHASE - set(ph.keys())
        if missing:
            fail(f"phases[{i}] missing keys: {sorted(missing)}")
        if not UPPER_SNAKE.match(ph["name"]):
            fail(f"phases[{i}].name {ph['name']!r} not UPPER_SNAKE")
        if ph["name"] in phase_names:
            fail(f"duplicate phase name {ph['name']!r}")
        phase_names.add(ph["name"])
        if not isinstance(ph["order"], int) or ph["order"] < 0:
            fail(f"phases[{i}].order must be a non-negative int")
        phase_orders.append(ph["order"])
    if len(set(phase_orders)) != len(phase_orders):
        fail(f"duplicate phase orders: {sorted(phase_orders)}")

    # Atoms
    atoms = d.get("atoms", [])
    if not isinstance(atoms, list):
        fail("atoms must be a list")
    if len(atoms) < 5:
        fail(f"atoms cardinality {len(atoms)} < 5 minimum")
    atom_names = set()
    for i, a in enumerate(atoms):
        missing = REQUIRED_ATOM - set(a.keys())
        if missing:
            fail(f"atoms[{i}] missing keys: {sorted(missing)}")
        if a["name"] in atom_names:
            fail(f"duplicate atom name {a['name']!r}")
        atom_names.add(a["name"])
        if a["phase"] not in phase_names:
            fail(f"atoms[{i}].phase {a['phase']!r} not in phases")
        if not isinstance(a["inputs"], list) or not isinstance(a["outputs"], list):
            fail(f"atoms[{i}].inputs and outputs must be lists")
        for port_ref in a["inputs"] + a["outputs"]:
            if port_ref not in port_names:
                fail(f"atoms[{i}={a['name']!r}] references undefined port_type {port_ref!r}")

    print(f"VALID: {p.name} — {len(port_types)} port_types, {len(phases)} phases, {len(atoms)} atoms; referential integrity OK")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_decisions_schema.py <decisions_json_path>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
