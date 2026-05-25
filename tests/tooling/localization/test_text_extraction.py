"""Tests for text extraction."""

import pytest
from engine.tooling.localization.text_extraction import (
    ExtractionSource,
    ExtractedString,
    ExtractionPattern,
    CodeExtractor,
    AssetExtractor,
    TextExtractionTool,
)


class TestExtractedString:
    """Tests for extracted string."""

    def test_creation(self):
        """Test extracted string creation."""
        string = ExtractedString(
            text="Hello",
            source_file="test.py",
            source_line=10,
            source_type=ExtractionSource.CODE_PYTHON,
        )
        assert string.text == "Hello"
        assert string.source_file == "test.py"
        assert string.source_line == 10

    def test_generate_key_from_text(self):
        """Test key generation from text."""
        string = ExtractedString(
            text="Hello World",
            source_file="test.py",
            source_line=1,
            source_type=ExtractionSource.CODE_PYTHON,
        )
        key = string.generate_key()
        assert "hello" in key.lower()
        assert "world" in key.lower()

    def test_generate_key_with_prefix(self):
        """Test key generation with prefix."""
        string = ExtractedString(
            text="Hello",
            source_file="test.py",
            source_line=1,
            source_type=ExtractionSource.CODE_PYTHON,
        )
        key = string.generate_key("ui.buttons")
        assert key.startswith("ui.buttons.")

    def test_generate_key_with_suggestion(self):
        """Test key generation uses suggestion."""
        string = ExtractedString(
            text="Hello",
            source_file="test.py",
            source_line=1,
            source_type=ExtractionSource.CODE_PYTHON,
            key_suggestion="my.custom.key",
        )
        key = string.generate_key()
        assert key == "my.custom.key"

    def test_truncate_long_key(self):
        """Test long keys are truncated."""
        string = ExtractedString(
            text="This is a very long string that should be truncated when generating a key",
            source_file="test.py",
            source_line=1,
            source_type=ExtractionSource.CODE_PYTHON,
        )
        key = string.generate_key()
        assert len(key) <= 50


class TestExtractionPattern:
    """Tests for extraction pattern."""

    def test_creation(self):
        """Test pattern creation."""
        pattern = ExtractionPattern(
            name="test",
            regex=r'_\("(.+?)"\)',
            source_type=ExtractionSource.CODE_PYTHON,
        )
        assert pattern.name == "test"
        assert pattern.group_index == 1


class TestCodeExtractor:
    """Tests for code extractor."""

    def setup_method(self):
        """Set up test extractor."""
        self.extractor = CodeExtractor()

    def test_extract_python_gettext(self):
        """Test extracting Python gettext strings."""
        code = '''
message = _("Hello World")
label = _('Click me')
'''
        results = self.extractor.extract_from_content(
            code, "test.py", ExtractionSource.CODE_PYTHON
        )
        texts = [r.text for r in results]
        assert "Hello World" in texts
        assert "Click me" in texts

    def test_extract_python_localize(self):
        """Test extracting Python localize strings."""
        code = '''
text = localize("Welcome")
'''
        results = self.extractor.extract_from_content(
            code, "test.py", ExtractionSource.CODE_PYTHON
        )
        texts = [r.text for r in results]
        assert "Welcome" in texts

    def test_extract_python_tr(self):
        """Test extracting Python tr strings."""
        code = '''
msg = tr("Goodbye")
'''
        results = self.extractor.extract_from_content(
            code, "test.py", ExtractionSource.CODE_PYTHON
        )
        texts = [r.text for r in results]
        assert "Goodbye" in texts

    def test_extract_cpp_tr(self):
        """Test extracting C++ TR strings."""
        code = '''
QString msg = TR("Hello C++");
'''
        results = self.extractor.extract_from_content(
            code, "test.cpp", ExtractionSource.CODE_CPP
        )
        texts = [r.text for r in results]
        assert "Hello C++" in texts

    def test_extract_cpp_loctext(self):
        """Test extracting C++ LOCTEXT strings."""
        code = '''
FText Text = LOCTEXT("ButtonLabel", "Click Here");
'''
        results = self.extractor.extract_from_content(
            code, "test.cpp", ExtractionSource.CODE_CPP
        )
        texts = [r.text for r in results]
        assert "Click Here" in texts

    def test_extract_javascript(self):
        """Test extracting JavaScript strings."""
        code = '''
const msg = t("Hello JS");
const msg2 = i18n("Localized");
'''
        results = self.extractor.extract_from_content(
            code, "test.js", ExtractionSource.CODE_JAVASCRIPT
        )
        texts = [r.text for r in results]
        assert "Hello JS" in texts
        assert "Localized" in texts

    def test_line_numbers(self):
        """Test line numbers are correct."""
        code = '''line1
line2
message = _("Hello")
line4
'''
        results = self.extractor.extract_from_content(
            code, "test.py", ExtractionSource.CODE_PYTHON
        )
        assert results[0].source_line == 3

    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        pattern = ExtractionPattern(
            name="custom",
            regex=r'LOC\("(.+?)"\)',
            source_type=ExtractionSource.CODE_PYTHON,
        )
        self.extractor.add_pattern(pattern)

        code = 'msg = LOC("Custom")'
        results = self.extractor.extract_from_content(
            code, "test.py", ExtractionSource.CODE_PYTHON
        )
        texts = [r.text for r in results]
        assert "Custom" in texts

    def test_detect_source_type(self):
        """Test automatic source type detection."""
        results = self.extractor.extract_from_content(
            '_("test")', "test.py"
        )
        assert results[0].source_type == ExtractionSource.CODE_PYTHON


class TestAssetExtractor:
    """Tests for asset extractor."""

    def setup_method(self):
        """Set up test extractor."""
        self.extractor = AssetExtractor()

    def test_extract_json_text(self):
        """Test extracting from JSON."""
        json_content = '''
{
    "label": "Hello",
    "description": "World",
    "data": 123
}
'''
        results = self.extractor.extract_from_json(json_content, "test.json")
        texts = [r.text for r in results]
        assert "Hello" in texts
        assert "World" in texts

    def test_extract_json_nested(self):
        """Test extracting nested JSON."""
        json_content = '''
{
    "button": {
        "text": "Click Me",
        "tooltip": "Do something"
    }
}
'''
        results = self.extractor.extract_from_json(json_content, "test.json")
        texts = [r.text for r in results]
        assert "Click Me" in texts
        assert "Do something" in texts

    def test_extract_xml(self):
        """Test extracting from XML."""
        xml_content = '''
<UI>
    <Text>Hello XML</Text>
    <Label>World</Label>
</UI>
'''
        results = self.extractor.extract_from_xml(xml_content, "test.xml")
        texts = [r.text for r in results]
        assert "Hello XML" in texts
        assert "World" in texts

    def test_add_json_key(self):
        """Test adding custom JSON key."""
        self.extractor.add_json_key("custom_text")

        json_content = '''
{
    "custom_text": "Custom Value"
}
'''
        results = self.extractor.extract_from_json(json_content, "test.json")
        texts = [r.text for r in results]
        assert "Custom Value" in texts


class TestTextExtractionTool:
    """Tests for unified extraction tool."""

    def setup_method(self):
        """Set up test tool."""
        self.tool = TextExtractionTool()

    def test_creation(self):
        """Test tool creation."""
        assert self.tool.code_extractor is not None
        assert self.tool.asset_extractor is not None

    def test_get_extracted_strings(self):
        """Test getting extracted strings."""
        # Initially empty
        strings = self.tool.get_extracted_strings()
        assert len(strings) == 0

    def test_get_strings_by_file(self):
        """Test getting strings by file."""
        strings = self.tool.get_strings_by_file()
        assert isinstance(strings, dict)

    def test_generate_string_table_entries(self):
        """Test generating string table entries."""
        entries = self.tool.generate_string_table_entries("ui")
        assert isinstance(entries, list)

    def test_export_json_format(self):
        """Test exporting in JSON format."""
        export = self.tool.export_for_translation("json")
        assert isinstance(export, str)

    def test_export_csv_format(self):
        """Test exporting in CSV format."""
        export = self.tool.export_for_translation("csv")
        assert isinstance(export, str)
        assert "key,text" in export
