import pytest

from dbs_vector.infrastructure.chunking.filters import FilterRegistry


def test_resolve_empty_returns_no_filters():
    assert FilterRegistry.resolve([]) == []


def test_unknown_filter_name_raises():
    with pytest.raises(ValueError, match="Unknown exclusion filter"):
        FilterRegistry.resolve(["does_not_exist"])


def test_excalidraw_skips_excalidraw_files():
    flt = FilterRegistry.resolve(["excalidraw"])[0]
    assert flt.should_skip_file("notes/Diagram.excalidraw.md", "anything") is True
    assert flt.should_skip_file("notes/Real.md", "# Title") is False


def test_excalidraw_skips_files_with_plugin_frontmatter():
    flt = FilterRegistry.resolve(["excalidraw"])[0]
    content = "---\nexcalidraw-plugin: parsed\n---\n# Drawing\n"
    assert flt.should_skip_file("notes/Real.md", content) is True


def test_excalidraw_drops_excalidraw_json_block():
    flt = FilterRegistry.resolve(["excalidraw"])[0]
    block = '{\n  "type": "excalidraw",\n  "version": 2\n}'
    assert flt.should_drop_block(block, "json") is True
    assert flt.should_drop_block("normal text", None) is False


def test_compressed_json_drops_by_info_string():
    flt = FilterRegistry.resolve(["compressed_json"])[0]
    assert flt.should_drop_block("N4KAkARAL...", "compressed-json") is True
    assert flt.should_drop_block("print('hi')", "python") is False
