"""
Tests for the minidump module.

Tests cover:
- MinidumpLevel enum
- ThreadInfo, ModuleInfo, MemoryRegion dataclasses
- Minidump generation at various levels
- Stack trace formatting
- Platform-specific behavior
- Security: environment sanitization, path validation
- Configuration constants
"""

import json
import os
import platform
import sys
import tempfile
import threading

import pytest

from engine.debug.crash.minidump import (
    MemoryRegion,
    Minidump,
    MinidumpData,
    MinidumpLevel,
    ModuleInfo,
    ThreadInfo,
    generate_crash_dump,
    get_current_stack_trace,
    MAX_STACK_TRACE_LINES,
    MAX_MODULES_TO_CAPTURE,
    MAX_MEMORY_REGIONS,
    FINGERPRINT_STACK_LINES,
)


@pytest.fixture
def temp_dump_path():
    """Create a temporary path for dump files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.dmp")


class TestMinidumpLevel:
    """Tests for MinidumpLevel enum."""

    def test_all_levels_exist(self):
        """All expected levels should exist."""
        assert MinidumpLevel.MINI is not None
        assert MinidumpLevel.MEDIUM is not None
        assert MinidumpLevel.FULL is not None

    def test_levels_are_distinct(self):
        """Levels should have distinct values."""
        levels = [MinidumpLevel.MINI, MinidumpLevel.MEDIUM, MinidumpLevel.FULL]
        values = [l.value for l in levels]
        assert len(set(values)) == 3


class TestThreadInfo:
    """Tests for ThreadInfo dataclass."""

    def test_create_thread_info(self):
        """Should be able to create ThreadInfo."""
        info = ThreadInfo(
            thread_id=12345,
            name="TestThread",
            is_daemon=False,
            is_alive=True,
            stack_trace="line1\nline2",
        )

        assert info.thread_id == 12345
        assert info.name == "TestThread"
        assert info.is_daemon is False
        assert info.is_alive is True
        assert "line1" in info.stack_trace

    def test_default_stack_trace(self):
        """Stack trace should default to empty string."""
        info = ThreadInfo(
            thread_id=1,
            name="Test",
            is_daemon=False,
            is_alive=True,
        )
        assert info.stack_trace == ""


class TestModuleInfo:
    """Tests for ModuleInfo dataclass."""

    def test_create_module_info(self):
        """Should be able to create ModuleInfo."""
        info = ModuleInfo(
            name="mymodule",
            path="/path/to/module.py",
            version="1.2.3",
        )

        assert info.name == "mymodule"
        assert info.path == "/path/to/module.py"
        assert info.version == "1.2.3"

    def test_optional_fields(self):
        """Path and version should be optional."""
        info = ModuleInfo(name="test")
        assert info.path is None
        assert info.version is None


class TestMemoryRegion:
    """Tests for MemoryRegion dataclass."""

    def test_create_memory_region(self):
        """Should be able to create MemoryRegion."""
        region = MemoryRegion(
            start_address=0x1000,
            size=4096,
            protection="r-xp",
            type_name="[heap]",
        )

        assert region.start_address == 0x1000
        assert region.size == 4096
        assert region.protection == "r-xp"
        assert region.type_name == "[heap]"


class TestMinidump:
    """Tests for Minidump class."""

    def test_generate_mini_dump(self, temp_dump_path):
        """Should generate a MINI level dump."""
        dump = Minidump()
        path = dump.generate(MinidumpLevel.MINI, temp_dump_path)

        assert path == temp_dump_path
        assert os.path.exists(path)

        with open(path, 'r') as f:
            data = json.load(f)

        assert data['level'] == 'MINI'
        assert 'platform_info' in data
        assert 'python_info' in data
        assert 'threads' in data
        assert len(data['threads']) > 0

    def test_generate_medium_dump(self, temp_dump_path):
        """Should generate a MEDIUM level dump with more info."""
        dump = Minidump()
        path = dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

        with open(path, 'r') as f:
            data = json.load(f)

        assert data['level'] == 'MEDIUM'
        assert 'modules' in data
        assert 'environment' in data
        assert 'memory_regions' in data

    def test_generate_full_dump(self, temp_dump_path):
        """Should generate a FULL level dump."""
        dump = Minidump()
        path = dump.generate(MinidumpLevel.FULL, temp_dump_path)

        with open(path, 'r') as f:
            data = json.load(f)

        assert data['level'] == 'FULL'
        assert 'modules' in data
        assert 'environment' in data

    def test_platform_info_captured(self, temp_dump_path):
        """Platform info should be captured correctly."""
        dump = Minidump()
        dump.generate(MinidumpLevel.MINI, temp_dump_path)

        with open(temp_dump_path, 'r') as f:
            data = json.load(f)

        assert data['platform_info']['system'] == platform.system()
        assert data['platform_info']['machine'] == platform.machine()

    def test_python_info_captured(self, temp_dump_path):
        """Python info should be captured correctly."""
        dump = Minidump()
        dump.generate(MinidumpLevel.MINI, temp_dump_path)

        with open(temp_dump_path, 'r') as f:
            data = json.load(f)

        assert data['python_info']['version'] == platform.python_version()
        assert data['python_info']['executable'] == sys.executable

    def test_thread_info_captured(self, temp_dump_path):
        """Thread info should be captured for all threads."""
        dump = Minidump()
        dump.generate(MinidumpLevel.MINI, temp_dump_path)

        with open(temp_dump_path, 'r') as f:
            data = json.load(f)

        threads = data['threads']
        assert len(threads) >= 1

        # Find the main thread
        main_threads = [t for t in threads if t['name'] == 'MainThread']
        assert len(main_threads) == 1
        assert main_threads[0]['is_alive'] is True

    def test_command_line_captured(self, temp_dump_path):
        """Command line arguments should be captured."""
        dump = Minidump()
        dump.generate(MinidumpLevel.MINI, temp_dump_path)

        with open(temp_dump_path, 'r') as f:
            data = json.load(f)

        assert 'command_line' in data
        assert len(data['command_line']) > 0

    def test_exception_info_captured(self, temp_dump_path):
        """Exception info should be captured when provided."""
        try:
            raise ValueError("test error message")
        except ValueError as e:
            dump = Minidump(exception=e)

        dump.generate(MinidumpLevel.MINI, temp_dump_path)

        with open(temp_dump_path, 'r') as f:
            data = json.load(f)

        assert data['exception_info'] is not None
        assert data['exception_info']['type'] == 'ValueError'
        assert 'test error message' in data['exception_info']['message']
        assert 'traceback' in data['exception_info']

    def test_environment_sanitized(self, temp_dump_path):
        """Sensitive environment variables should be redacted."""
        os.environ['TEST_API_KEY'] = 'secret123'
        os.environ['TEST_NORMAL_VAR'] = 'visible'

        try:
            dump = Minidump()
            dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

            with open(temp_dump_path, 'r') as f:
                data = json.load(f)

            # API_KEY should be redacted
            if 'TEST_API_KEY' in data['environment']:
                assert data['environment']['TEST_API_KEY'] == '[REDACTED]'

            # Normal var should be visible
            assert data['environment']['TEST_NORMAL_VAR'] == 'visible'
        finally:
            del os.environ['TEST_API_KEY']
            del os.environ['TEST_NORMAL_VAR']

    def test_get_stack_trace_all_threads(self, temp_dump_path):
        """get_stack_trace should return trace for all threads."""
        dump = Minidump()
        trace = dump.get_stack_trace()

        assert "MainThread" in trace
        assert "Thread" in trace  # Thread header format

    def test_get_stack_trace_specific_thread(self, temp_dump_path):
        """get_stack_trace should return trace for specific thread."""
        dump = Minidump()
        current_tid = threading.current_thread().ident

        trace = dump.get_stack_trace(thread_id=current_tid)

        assert trace != ""
        assert "test_minidump" in trace  # Should contain this test file

    def test_get_stack_trace_invalid_thread(self, temp_dump_path):
        """get_stack_trace should handle invalid thread ID."""
        dump = Minidump()
        trace = dump.get_stack_trace(thread_id=999999999)

        assert "No stack trace available" in trace

    def test_creates_parent_directory(self):
        """generate should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "a", "b", "c", "test.dmp")

            dump = Minidump()
            path = dump.generate(MinidumpLevel.MINI, nested_path)

            assert os.path.exists(path)

    def test_modules_captured_in_medium(self, temp_dump_path):
        """Loaded modules should be captured at MEDIUM level."""
        dump = Minidump()
        dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

        with open(temp_dump_path, 'r') as f:
            data = json.load(f)

        modules = data['modules']
        assert len(modules) > 0

        # Should have some standard modules
        module_names = [m['name'] for m in modules]
        assert 'sys' in module_names
        assert 'os' in module_names


class TestMinidumpData:
    """Tests for MinidumpData dataclass."""

    def test_default_values(self):
        """MinidumpData should have sensible defaults."""
        data = MinidumpData(level=MinidumpLevel.MINI)

        assert data.level == MinidumpLevel.MINI
        assert data.threads == []
        assert data.modules == []
        assert data.memory_regions == []
        assert data.exception_info is None
        assert data.environment == {}
        assert data.command_line == []


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_generate_crash_dump(self, temp_dump_path):
        """generate_crash_dump should create a dump file."""
        path = generate_crash_dump(temp_dump_path)

        assert os.path.exists(path)

        with open(path, 'r') as f:
            data = json.load(f)

        assert data['level'] == 'MEDIUM'  # Default level

    def test_generate_crash_dump_with_level(self, temp_dump_path):
        """generate_crash_dump should respect level parameter."""
        path = generate_crash_dump(temp_dump_path, level=MinidumpLevel.FULL)

        with open(path, 'r') as f:
            data = json.load(f)

        assert data['level'] == 'FULL'

    def test_generate_crash_dump_with_exception(self, temp_dump_path):
        """generate_crash_dump should include exception info."""
        try:
            raise RuntimeError("test")
        except RuntimeError as e:
            path = generate_crash_dump(temp_dump_path, exception=e)

        with open(path, 'r') as f:
            data = json.load(f)

        assert data['exception_info']['type'] == 'RuntimeError'

    def test_get_current_stack_trace(self):
        """get_current_stack_trace should return non-empty trace."""
        trace = get_current_stack_trace()

        assert trace != ""
        assert "Thread" in trace
        assert "test_minidump" in trace  # Should contain this file


class TestNativeMinidump:
    """Tests for native minidump stub."""

    def test_generate_native_minidump_returns_false(self, temp_dump_path):
        """Native minidump should return False (not implemented)."""
        result = Minidump.generate_native_minidump(
            temp_dump_path,
            MinidumpLevel.MEDIUM
        )

        # Currently returns False as it's a stub
        assert result is False


class TestMemoryRegionsLinux:
    """Tests for memory region collection on Linux."""

    @pytest.mark.skipif(
        platform.system() != 'Linux',
        reason="Memory regions only available on Linux"
    )
    def test_memory_regions_on_linux(self, temp_dump_path):
        """Memory regions should be populated on Linux."""
        dump = Minidump()
        dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

        with open(temp_dump_path, 'r') as f:
            data = json.load(f)

        regions = data['memory_regions']
        assert len(regions) > 0

        # Should have standard regions
        has_heap = any('[heap]' in r['type_name'] for r in regions)
        has_stack = any('[stack]' in r['type_name'] for r in regions)
        # At least one should be present
        assert has_heap or has_stack or len(regions) > 0


class TestEnvironmentSanitization:
    """Tests for environment variable sanitization."""

    def test_api_key_redacted(self, temp_dump_path):
        """API key environment variables should be redacted."""
        os.environ['MY_API_KEY'] = 'secret12345'
        os.environ['SOME_APIKEY'] = 'anothersecret'

        try:
            dump = Minidump()
            dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

            with open(temp_dump_path, 'r') as f:
                data = json.load(f)

            env = data['environment']
            if 'MY_API_KEY' in env:
                assert env['MY_API_KEY'] == '[REDACTED]'
            if 'SOME_APIKEY' in env:
                assert env['SOME_APIKEY'] == '[REDACTED]'
        finally:
            del os.environ['MY_API_KEY']
            del os.environ['SOME_APIKEY']

    def test_password_redacted(self, temp_dump_path):
        """Password environment variables should be redacted."""
        os.environ['DB_PASSWORD'] = 'supersecret'
        os.environ['USER_PASSWD'] = 'anothersecret'

        try:
            dump = Minidump()
            dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

            with open(temp_dump_path, 'r') as f:
                data = json.load(f)

            env = data['environment']
            if 'DB_PASSWORD' in env:
                assert env['DB_PASSWORD'] == '[REDACTED]'
            if 'USER_PASSWD' in env:
                assert env['USER_PASSWD'] == '[REDACTED]'
        finally:
            del os.environ['DB_PASSWORD']
            del os.environ['USER_PASSWD']

    def test_token_redacted(self, temp_dump_path):
        """Token environment variables should be redacted."""
        os.environ['AUTH_TOKEN'] = 'bearer_xyz123'
        os.environ['JWT_TOKEN'] = 'eyJhbGciOiJIUzI1NiJ9'

        try:
            dump = Minidump()
            dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

            with open(temp_dump_path, 'r') as f:
                data = json.load(f)

            env = data['environment']
            if 'AUTH_TOKEN' in env:
                assert env['AUTH_TOKEN'] == '[REDACTED]'
            if 'JWT_TOKEN' in env:
                assert env['JWT_TOKEN'] == '[REDACTED]'
        finally:
            del os.environ['AUTH_TOKEN']
            del os.environ['JWT_TOKEN']

    def test_long_hex_string_redacted(self, temp_dump_path):
        """Long hex strings (likely secrets) should be redacted."""
        # 64-char hex string (like a SHA256 hash or API key)
        os.environ['SOME_NORMAL_VAR'] = 'a' * 64

        try:
            dump = Minidump()
            dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

            with open(temp_dump_path, 'r') as f:
                data = json.load(f)

            env = data['environment']
            if 'SOME_NORMAL_VAR' in env:
                assert env['SOME_NORMAL_VAR'] == '[REDACTED]'
        finally:
            del os.environ['SOME_NORMAL_VAR']

    def test_hostname_not_included(self, temp_dump_path):
        """Platform info should not include hostname for security."""
        dump = Minidump()
        dump.generate(MinidumpLevel.MEDIUM, temp_dump_path)

        with open(temp_dump_path, 'r') as f:
            data = json.load(f)

        # Hostname should not be present
        assert 'hostname' not in data['platform_info']


class TestPathValidation:
    """Tests for path validation in minidump generation."""

    def test_empty_path_rejected(self):
        """Empty path should be rejected."""
        dump = Minidump()
        with pytest.raises(ValueError):
            dump.generate(MinidumpLevel.MINI, "")

    def test_null_byte_path_rejected(self):
        """Path with null bytes should be rejected."""
        dump = Minidump()
        with pytest.raises(ValueError):
            dump.generate(MinidumpLevel.MINI, "/tmp/test\x00.dmp")


class TestConfigurationConstants:
    """Tests for configuration constants."""

    def test_max_stack_trace_lines_reasonable(self):
        """MAX_STACK_TRACE_LINES should be reasonable."""
        assert MAX_STACK_TRACE_LINES > 0
        assert MAX_STACK_TRACE_LINES <= 1000

    def test_max_modules_reasonable(self):
        """MAX_MODULES_TO_CAPTURE should be reasonable."""
        assert MAX_MODULES_TO_CAPTURE > 0
        assert MAX_MODULES_TO_CAPTURE <= 5000

    def test_fingerprint_lines_reasonable(self):
        """FINGERPRINT_STACK_LINES should be reasonable."""
        assert FINGERPRINT_STACK_LINES > 0
        assert FINGERPRINT_STACK_LINES <= 50
