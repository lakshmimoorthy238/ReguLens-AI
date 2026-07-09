import json
import re
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "references" / "parsed" / "guideline_pages.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "references" / "parsed" / "guideline_chunks.jsonl"

# Chunk size is in characters, not tokens.
# This is simple and good enough for MVP.
MAX_CHARS = 1200
OVERLAP_CHARS = 200
MIN_CHARS = 80

# How far into a chunk we look for a section header. Kept small-ish so we're
# mostly catching headers near the top of a chunk, not incidental
# cross-references to other sections/guidelines buried in body text.
SECTION_SEARCH_CHARS = 500

SECTION_PATTERNS = [
    # CTD sections like 3.2.P.8.2, 2.7.4, 3.2.P.8, 5.3.5.1
    r"\b\d+(?:\.\d+)+(?:\.[A-Z])?(?:\.\d+)*\b",
    # ICH guideline sections like Q1A, Q2(R2), M4Q
    r"\b[QEMS]\d+[A-Z]?(?:\(R\d+\))?\b",
]


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace without destroying readable structure."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str) -> Optional[str]:
    """
    Try to detect a CTD/ICH section identifier from the chunk text.

    This is not perfect, but useful metadata for retrieval and reporting.
    Only returns a match if found near the start of the chunk (see
    SECTION_SEARCH_CHARS) — this is a fresh detection, not inherited.
    """
    search_area = text[:SECTION_SEARCH_CHARS]

    for pattern in SECTION_PATTERNS:
        match = re.search(pattern, search_area)
        if match:
            return match.group(0)

    return None


def split_text_with_overlap(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
    """
    Split text into character-based chunks with overlap.

    Overlap helps RAG because important context near the boundary
    is not lost between chunks.
    """
    text = normalize_whitespace(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to end near a sentence or paragraph boundary.
        window = text[start:end]
        split_points = [
            window.rfind("\n\n"),
            window.rfind(". "),
            window.rfind("; "),
            window.rfind(": "),
        ]
        best_split = max(split_points)

        if best_split > max_chars * 0.5:
            end = start + best_split + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward with overlap.
        next_start = end - overlap_chars

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def make_chunk_id(record: dict, chunk_number: int) -> str:
    """Create stable chunk ID."""
    guideline_id = record.get("guideline_id", "UNKNOWN")
    version = record.get("version", "") or "NA"
    page_number = record.get("page_number", "NA")

    safe_version = (
        str(version)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

    return f"{guideline_id}_{safe_version}_p{page_number}_c{chunk_number}"


def chunk_page_record(record: dict, last_section: Optional[str]) -> tuple[list[dict], Optional[str]]:
    """
    Convert one page-level record into multiple chunk-level records.

    `last_section` is the most recently detected section identifier for
    this guideline, carried in from prior pages/chunks. If a chunk doesn't
    contain a fresh header, it inherits this value rather than being left
    blank — otherwise only the first chunk after each header would carry
    section metadata, breaking citation traceability for every chunk after
    it.

    Returns (chunk_records, updated_last_section).
    """
    text = record.get("text", "")
    char_count = record.get("char_count", len(text))
    possibly_scanned = record.get("possibly_scanned", False)

    # Skip empty or very tiny pages.
    if possibly_scanned or char_count < MIN_CHARS:
        return [], last_section

    text_chunks = split_text_with_overlap(text)

    chunk_records = []

    for idx, chunk_text in enumerate(text_chunks, start=1):
        if len(chunk_text) < MIN_CHARS:
            continue

        fresh_section = detect_section(chunk_text)
        if fresh_section:
            last_section = fresh_section
            section_confidence = "detected"
        elif last_section:
            section_confidence = "inherited"
        else:
            section_confidence = None

        chunk_record = {
            "chunk_id": make_chunk_id(record, idx),
            "guideline_id": record.get("guideline_id", ""),
            "title": record.get("title", ""),
            "version": record.get("version", ""),
            "domain": record.get("domain", ""),
            "source_file": record.get("source_file", ""),
            "source_hash": record.get("source_hash", ""),
            "index_version": record.get("index_version", ""),
            "page_number": record.get("page_number"),
            "chunk_number": idx,
            "detected_section": fresh_section or last_section,
            "fresh_section": fresh_section,
            "section_confidence": section_confidence,
            "text": chunk_text,
            "char_count": len(chunk_text),
        }

        chunk_records.append(chunk_record)

    return chunk_records, last_section


def chunk_guideline_pages(records: list[dict]) -> list[dict]:
    """
    Chunk all page-level guideline records.

    Records are sorted by (guideline_id, page_number) first to guarantee
    correct reading order — this matters because section inheritance
    depends on processing pages in the order they actually appear in the
    source document.
    """
    records = sorted(
        records,
        key=lambda r: (r.get("guideline_id", ""), r.get("page_number", 0)),
    )

    all_chunks = []
    last_section_by_guideline: dict[str, Optional[str]] = {}

    for record in records:
        guideline_id = record.get("guideline_id", "")
        last_section = last_section_by_guideline.get(guideline_id)

        page_chunks, last_section = chunk_page_record(record, last_section)

        last_section_by_guideline[guideline_id] = last_section
        all_chunks.extend(page_chunks)

    return all_chunks


def print_summary(chunks: list[dict]) -> None:
    """Print chunking summary."""
    counts = {}
    inherited_counts = {}
    no_section_counts = {}

    for chunk in chunks:
        guideline_id = chunk.get("guideline_id", "UNKNOWN")
        counts[guideline_id] = counts.get(guideline_id, 0) + 1

        confidence = chunk.get("section_confidence")
        if confidence == "inherited":
            inherited_counts[guideline_id] = inherited_counts.get(guideline_id, 0) + 1
        elif confidence is None:
            no_section_counts[guideline_id] = no_section_counts.get(guideline_id, 0) + 1

    print("\nGuideline Chunking Summary")
    print("--------------------------")
    print(f"Total chunks: {len(chunks)}")
    print()

    for guideline_id, count in counts.items():
        inherited = inherited_counts.get(guideline_id, 0)
        no_section = no_section_counts.get(guideline_id, 0)
        print(
            f"{guideline_id:12} | {count:4} chunks | "
            f"{inherited:4} inherited section | {no_section:4} no section"
        )


def main() -> None:
    page_records = load_jsonl(INPUT_PATH)
    chunks = chunk_guideline_pages(page_records)
    save_jsonl(chunks, OUTPUT_PATH)
    print_summary(chunks)
    print(f"\nSaved guideline chunks to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()