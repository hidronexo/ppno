from unittest.mock import mock_open, patch

import pytest

from ppno.section_parser import SectionParser


@pytest.fixture
def example_ext_file(tmp_path):
    path = tmp_path / "test.ext"
    content = """[SECTION1]
line1
line2

[SECTION2]
key1 = val1
key2 = val2
"""
    path.write_text(content)
    return path

def test_read_section(example_ext_file):
    reader = SectionParser(example_ext_file)
    sec1 = reader.read_section("SECTION1")
    assert len(sec1) == 2
    assert sec1[0] == (2, ("line1",))
    assert sec1[1] == (3, ("line2",))

def test_read_all(example_ext_file):
    reader = SectionParser(example_ext_file)
    sections = reader.read()
    assert "SECTION1" in sections
    assert "SECTION2" in sections
    assert len(sections["SECTION1"]) == 2

def test_read_rows_ignores_comments(tmp_path):
    path = tmp_path / "pipes.cat"
    path.write_text("; group diameter roughness price\nPVC 100 0.1 10 ; comment\n\nPVC 150 0.1 20\n")
    reader = SectionParser(path)

    assert reader.read_rows() == [
        (2, "PVC 100 0.1 10"),
        (4, "PVC 150 0.1 20"),
    ]

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        reader = SectionParser("nonexistent.ext")
        reader.read()

def test_different_encodings(tmp_path):
    # Test UTF-16
    path = tmp_path / "utf16.ext"
    path.write_text("[SECTION]\nvalue", encoding='utf-16')
    reader = SectionParser(path)
    sections = reader.read()
    assert sections["SECTION"][0] == (2, "value")

def test_read_lines_fallback(tmp_path):
    # Test latin-1 fallback by mocking failures in first 3 encodings
    path = tmp_path / "fallback.txt"
    path.write_text("test")
    parser = SectionParser(path)
    
    # Use the correct internal method name _get_lines
    with patch("pathlib.Path.open", side_effect=[UnicodeDecodeError("utf8", b"", 0, 1, ""), 
                                                UnicodeDecodeError("utf16", b"", 0, 1, ""),
                                                UnicodeDecodeError("cp1252", b"", 0, 1, ""),
                                                mock_open(read_data="FALLBACK").return_value]):
        lines = parser._get_lines()
        assert lines == ["FALLBACK"]

def test_line_to_tuple(tmp_path):
    # Use a real file path for init to avoid FileNotFoundError
    path = tmp_path / "dummy.txt"
    path.write_text("")
    parser = SectionParser(path)
    assert parser.line_to_tuple("a b c") == ("a", "b", "c")
    assert parser.line_to_tuple("a,b,c") == ("a", "b", "c")
    assert parser.line_to_tuple("  a   b  ") == ("a", "b")
    assert parser.line_to_tuple("") == ()
