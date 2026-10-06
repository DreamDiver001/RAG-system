"""Load markdown files from a docs folder and split them into retrievable chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source: str  # relative file path, for citing where an answer came from


def load_markdown_files(docs_dir: str) -> list[tuple[str, str]]:
    """Return list of (relative_path, file_content) for every .md file under docs_dir."""
    docs_path = Path(docs_dir)
    files = []
    for md_path in sorted(docs_path.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        rel_path = str(md_path.relative_to(docs_path))
        files.append((rel_path, text))
    return files


def chunk_markdown(text: str, source: str, max_chars: int = 1200) -> list[Chunk]:
    """Split markdown into chunks along heading boundaries, capping chunk size.

    Splitting on headings keeps each chunk topically coherent instead of
    cutting sentences in half at an arbitrary character offset.
    """
    sections = re.split(r"\n(?=#{1,6} )", text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(Chunk(text=section, source=source))
        else:
            # Section too big even on its own: fall back to paragraph splitting.
            paragraphs = section.split("\n\n")
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) + 2 <= max_chars:
                    buffer = f"{buffer}\n\n{para}" if buffer else para
                else:
                    if buffer:
                        chunks.append(Chunk(text=buffer, source=source))
                    buffer = para
            if buffer:
                chunks.append(Chunk(text=buffer, source=source))
    return chunks


def build_corpus(docs_dir: str) -> list[Chunk]:
    """Load all markdown files under docs_dir and return the full list of chunks."""
    all_chunks = []
    for rel_path, text in load_markdown_files(docs_dir):
        all_chunks.extend(chunk_markdown(text, rel_path))
    return all_chunks


if __name__ == "__main__":
    docs_dir = Path(__file__).resolve().parent.parent / "data" / "docs"
    chunks = build_corpus(str(docs_dir))
    print(f"{len(chunks)} chunks from markdown docs")
    print("--- sample chunk ---")
    print(f"source: {chunks[0].source}")
    print(chunks[0].text[:300])
