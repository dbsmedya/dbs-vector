from dbs_vector.core.ports import IContentFilter


class ExcalidrawFilter:
    name = "excalidraw"

    def should_skip_file(self, filepath: str, content: str) -> bool:
        if filepath.lower().endswith(".excalidraw.md"):
            return True
        head = content[:500]
        return "excalidraw-plugin" in head

    def should_drop_block(self, text: str, info_string: str | None) -> bool:
        if (info_string or "").strip().lower() == "json":
            return '"type": "excalidraw"' in text or '"type":"excalidraw"' in text
        return False


class CompressedJsonFilter:
    name = "compressed_json"

    def should_skip_file(self, filepath: str, content: str) -> bool:
        return False

    def should_drop_block(self, text: str, info_string: str | None) -> bool:
        return (info_string or "").strip().lower() == "compressed-json"


class GitignoreFilter:
    """Marker filter: `<root>/.gitignore` filters file DISCOVERY.

    Enforcement lives in `services.path_filter.PathFilter`, at the walk +
    watcher-event layer, before any file is read. This class exists so the
    name `gitignore` validates in `FilterRegistry` alongside the content
    filters; its chunk-level hooks are deliberate no-ops.
    """

    name = "gitignore"

    def should_skip_file(self, filepath: str, content: str) -> bool:
        return False

    def should_drop_block(self, text: str, info_string: str | None) -> bool:
        return False


class FilterRegistry:
    """Open/closed registry of named content filters (cf. ModelRegistry)."""

    _filters: dict[str, IContentFilter] = {
        ExcalidrawFilter.name: ExcalidrawFilter(),
        CompressedJsonFilter.name: CompressedJsonFilter(),
        GitignoreFilter.name: GitignoreFilter(),
    }

    @classmethod
    def register(cls, flt: IContentFilter) -> None:
        if flt.name in cls._filters:
            raise ValueError(f"Filter '{flt.name}' already registered")
        cls._filters[flt.name] = flt

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._filters)

    @classmethod
    def resolve(cls, names: list[str]) -> list[IContentFilter]:
        out: list[IContentFilter] = []
        for n in names:
            if n not in cls._filters:
                raise ValueError(f"Unknown exclusion filter '{n}'. Known: {cls.keys()}")
            out.append(cls._filters[n])
        return out
