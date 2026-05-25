#!/usr/bin/env python3
"""
Foundation Import System Tests

Verifies all 6 Foundation modules import correctly and exports are accessible.
"""

import sys
import traceback

# Ensure the project root is in the path
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

results = {
    'individual_imports': {'passed': False, 'error': None},
    'main_package_import': {'passed': False, 'error': None},
    'singleton_instances': {'passed': False, 'error': None},
    'cross_module_dependencies': {'passed': False, 'error': None},
}


def test_individual_imports():
    """Test 1: Individual Module Imports"""
    print("\n" + "="*60)
    print("TEST 1: Individual Module Imports")
    print("="*60)

    try:
        from foundation import mirror
        print("  - foundation.mirror: OK")

        from foundation import serializer
        print("  - foundation.serializer: OK")

        from foundation import registry
        print("  - foundation.registry: OK")

        from foundation import tracker
        print("  - foundation.tracker: OK")

        from foundation import inspector
        print("  - foundation.inspector: OK")

        from foundation import shell
        print("  - foundation.shell: OK")

        results['individual_imports']['passed'] = True
        print("\n[PASS] All individual imports successful")

    except Exception as e:
        results['individual_imports']['error'] = str(e)
        print(f"\n[FAIL] Import error: {e}")
        traceback.print_exc()
        raise


def test_main_package_import():
    """Test 2: Main Package Import - All Exports Accessible"""
    print("\n" + "="*60)
    print("TEST 2: Main Package Import")
    print("="*60)

    try:
        from foundation import (
            # Layer 0 - Mirror
            mirror, ObjectMirror, ClassMirror, FieldInfo, MethodInfo,
            STANDARD_METADATA_KEYS,
        )
        print("  - Layer 0 (Mirror): OK")

        from foundation import (
            # Serializer
            to_dict, from_dict, to_bytes, from_bytes, to_file, from_file,
            deep_copy, diff, patch, Delta, register_type,
            SerializationContext, DeserializationContext,
        )
        print("  - Layer 0 (Serializer): OK")

        from foundation import (
            # Layer 1
            registry, Registry,
        )
        print("  - Layer 1 (Registry): OK")

        from foundation import (
            # Layer 2
            tracker, Tracker, Change, Transaction, ChangeCallback,
        )
        print("  - Layer 2 (Tracker): OK")

        from foundation import (
            # Layer 3 - Inspector
            inspector, Inspector, InspectorPanel, View,
            UIContext, TextUIContext, FieldsView, RawView, JSONView,
            CollectionView, HistoryEntry,
        )
        print("  - Layer 3 (Inspector): OK")

        from foundation import (
            # Layer 3 - Shell
            shell, Shell, ExecutionResult, inspect,
        )
        print("  - Layer 3 (Shell): OK")

        results['main_package_import']['passed'] = True
        print("\n[PASS] All exports accessible")

    except Exception as e:
        results['main_package_import']['error'] = str(e)
        print(f"\n[FAIL] Export error: {e}")
        traceback.print_exc()
        raise


def test_singleton_instances():
    """Test 3: Singleton Instances Exist"""
    print("\n" + "="*60)
    print("TEST 3: Singleton Instances")
    print("="*60)

    try:
        from foundation import registry, tracker, inspector, shell

        assert registry is not None, "registry singleton missing"
        print(f"  - registry: {type(registry).__name__} (OK)")

        assert tracker is not None, "tracker singleton missing"
        print(f"  - tracker: {type(tracker).__name__} (OK)")

        assert inspector is not None, "inspector singleton missing"
        print(f"  - inspector: {type(inspector).__name__} (OK)")

        assert shell is not None, "shell singleton missing"
        print(f"  - shell: {type(shell).__name__} (OK)")

        results['singleton_instances']['passed'] = True
        print("\n[PASS] All singletons initialized")

    except AssertionError as e:
        results['singleton_instances']['error'] = str(e)
        print(f"\n[FAIL] Assertion error: {e}")
        raise
    except Exception as e:
        results['singleton_instances']['error'] = str(e)
        print(f"\n[FAIL] Error: {e}")
        traceback.print_exc()
        raise


def test_cross_module_dependencies():
    """Test 4: Cross-Module Dependencies"""
    print("\n" + "="*60)
    print("TEST 4: Cross-Module Dependencies")
    print("="*60)

    try:
        from foundation import shell, mirror, registry, tracker, inspector

        # Shell should have all systems in namespace
        ns = shell.namespace
        print(f"  Shell namespace contains {len(ns)} items")

        # Check required items in shell namespace
        required_items = ['mirror', 'registry', 'tracker', 'inspector']
        missing = []

        for item in required_items:
            if item in ns:
                print(f"  - '{item}' in shell namespace: OK")
            else:
                missing.append(item)
                print(f"  - '{item}' in shell namespace: MISSING")

        if missing:
            results['cross_module_dependencies']['error'] = f"Missing items: {missing}"
            print(f"\n[FAIL] Missing namespace items: {missing}")
            raise AssertionError(f"Missing namespace items: {missing}")

        # Verify the namespace items are the actual singleton instances
        print("\n  Verifying namespace items are correct instances:")
        from foundation import mirror as mirror_module

        # The 'mirror' in namespace should be the mirror function
        if callable(ns.get('mirror')):
            print("  - mirror: callable (OK)")
        else:
            print("  - mirror: not callable (WARNING)")

        # Check registry, tracker, inspector are the singletons
        if ns.get('registry') is registry:
            print("  - registry: same instance (OK)")
        else:
            print("  - registry: different instance (WARNING)")

        if ns.get('tracker') is tracker:
            print("  - tracker: same instance (OK)")
        else:
            print("  - tracker: different instance (WARNING)")

        if ns.get('inspector') is inspector:
            print("  - inspector: same instance (OK)")
        else:
            print("  - inspector: different instance (WARNING)")

        results['cross_module_dependencies']['passed'] = True
        print("\n[PASS] Shell namespace properly populated")

    except Exception as e:
        results['cross_module_dependencies']['error'] = str(e)
        print(f"\n[FAIL] Error: {e}")
        traceback.print_exc()
        raise


def print_summary():
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = len(results)
    passed = sum(1 for r in results.values() if r['passed'])

    for name, result in results.items():
        status = "PASS" if result['passed'] else "FAIL"
        error_msg = f" - {result['error']}" if result['error'] else ""
        print(f"  [{status}] {name}{error_msg}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*60)

    return passed == total


if __name__ == "__main__":
    print("Foundation Import System Tests")
    print("Python version:", sys.version)

    # Run all tests
    test_individual_imports()
    test_main_package_import()
    test_singleton_instances()
    test_cross_module_dependencies()

    # Print summary
    all_passed = print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
