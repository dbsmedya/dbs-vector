from collections.abc import Iterator

import markdown_it

from dbs_vector.core.models import Chunk, Document


class DocumentChunker:
    """
    Robust Chunker: Splits prose and code natively.
    Uses markdown-it-py for precise semantic parsing of markdown files.
    Falls back to naive text splitting for text files.
    """

    def __init__(self, max_chars: int = 1000) -> None:
        self.max_chars = max_chars
        self.md_parser = markdown_it.MarkdownIt()

    def process(self, document: Document) -> Iterator[Chunk]:
        """Yields chunks from a raw document."""
        if document.filepath.lower().endswith(".md"):
            yield from self._chunk_markdown(document)
        else:
            yield from self._chunk_text(document)

    def _chunk_markdown(self, document: Document) -> Iterator[Chunk]:
        """Chunks markdown by grouping top-level semantic tokens (headings, paragraphs, code blocks)."""
        tokens = self.md_parser.parse(document.content)
        lines = document.content.splitlines(keepends=True)

        chunks_text: list[str] = []
        current_chunk_text = ""

        # We iterate over top-level semantic blocks
        for token in tokens:
            if token.level == 0 and token.map is not None:
                start_line, end_line = token.map
                block_text = "".join(lines[start_line:end_line])

                # If this block is a code fence, keep it atomic regardless of size.
                if token.type == "fence":
                    if current_chunk_text.strip():
                        chunks_text.append(current_chunk_text.strip())
                        current_chunk_text = ""
                    chunks_text.append(block_text.strip())
                else:
                    # Accumulate prose up to max_chars limit
                    if (
                        len(current_chunk_text) + len(block_text) > self.max_chars
                        and current_chunk_text.strip()
                    ):
                        chunks_text.append(current_chunk_text.strip())
                        current_chunk_text = block_text
                    else:
                        if current_chunk_text:
                            current_chunk_text += "\n\n" + block_text.strip()
                        else:
                            current_chunk_text = block_text.strip()

        if current_chunk_text.strip():
            chunks_text.append(current_chunk_text.strip())

        yield from self._create_chunks(document, chunks_text)

    def _chunk_text(self, document: Document) -> Iterator[Chunk]:
        """Fallback simple chunking for .txt files by splitting on double newlines."""
        paragraphs = document.content.split("\n\n")

        chunks_text: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) > self.max_chars and current:
                chunks_text.append(current.strip())
                current = paragraph
            else:
                if current:
                    current += "\n\n" + paragraph
                else:
                    current = paragraph

        if current.strip():
            chunks_text.append(current.strip())

        yield from self._create_chunks(document, chunks_text)

    def _create_chunks(self, document: Document, texts: list[str]) -> Iterator[Chunk]:
        """Helper to yield final Chunk objects, filtering noise."""
        valid_texts = [text for text in texts if len(text.strip()) >= 5]

        for i, text in enumerate(valid_texts):
            yield Chunk(
                id=f"{document.filepath}_chunk_{i}",
                text=text,
                source=document.filepath,
                content_hash=document.content_hash,
            )
