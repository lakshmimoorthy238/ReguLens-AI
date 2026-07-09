import json
import re
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT /"references"/ "parsed" / "dossier_pages.jsonl"
OUTPUT_PATH = PROJECT_ROOT /"references"/ "parsed" / "dossier_chunks.jsonl"

MAX_CHARS = 1000
OVERLAP_CHARS = 150
MIN_CHARS = 50
SECTION_SEARCH_CHARS = 500

# Matches CTD-style dotted section codes: 3.2.P.8.2, 2.7.4, etc.
# Requires at least two levels (a bare "7" won't match this on its own).
CTD_SECTION_PATTERN = re.compile(
    r"\b\d+(?:\.(?:\d+|[A-Z]))+\b",
    flags=re.IGNORECASE,
)

# Fallback for documents that use simple top-level numbered headings instead
# of CTD dotted codes — e.g. ICH E3-style Clinical Study Reports use
# "1. Ethics", "7. Efficacy Evaluation" rather than 3.2.P.8-style numbers.
# Only tried if CTD_SECTION_PATTERN finds nothing, and only matches whole
# heading-looking lines (not incidental numbers mid-sentence).
SIMPLE_HEADING_PATTERN = re.compile(
    r"^(\d{1,2}\.\s+[A-Z][A-Za-z0-9 ,&/\-]{2,80})$",
    flags=re.MULTILINE,
)


def load_jsonl(path: Path) -> list[dict]:
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
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str) -> Optional[str]:
    search_area = text[:SECTION_SEARCH_CHARS]

    match = CTD_SECTION_PATTERN.search(search_area)
    if match:
        return match.group(0)

    fallback_match = SIMPLE_HEADING_PATTERN.search(search_area)
    if fallback_match:
        return fallback_match.group(0).strip()

    return None


def split_text_with_overlap(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
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

        next_start = end - overlap_chars

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def make_chunk_id(record: dict, chunk_number: int) -> str:
    dossier_id = record.get("dossier_id", "dossier")
    file_stem = Path(record.get("file_name", "unknown")).stem
    page_number = record.get("page_number", "NA")

    safe_file_stem = re.sub(r"[^a-zA-Z0-9_]+", "_", file_stem)

    return f"{dossier_id}_{safe_file_stem}_p{page_number}_c{chunk_number}"


def chunk_page_record(
    record: dict,
    last_section: Optional[str],
) -> tuple[list[dict], Optional[str]]:
    text = record.get("text", "")
    char_count = record.get("char_count", len(text))
    possibly_scanned = record.get("possibly_scanned", False)

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
            "dossier_id": record.get("dossier_id", ""),
            "file_name": record.get("file_name", ""),
            "source_hash": record.get("source_hash", ""),
            "document_type": record.get("document_type", ""),
            "module_guess": record.get("module_guess", ""),
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


def chunk_dossier_pages(records: list[dict]) -> list[dict]:
    records = sorted(
        records,
        key=lambda r: (
            r.get("file_name", ""),
            r.get("page_number", 0),
        ),
    )

    all_chunks = []
    last_section_by_file: dict[str, Optional[str]] = {}

    for record in records:
        file_name = record.get("file_name", "")
        last_section = last_section_by_file.get(file_name)

        page_chunks, last_section = chunk_page_record(record, last_section)

        last_section_by_file[file_name] = last_section
        all_chunks.extend(page_chunks)

    return all_chunks


def print_summary(chunks: list[dict]) -> None:
    counts = {}
    inherited_counts = {}
    no_section_counts = {}

    for chunk in chunks:
        doc_type = chunk.get("document_type", "unknown")
        counts[doc_type] = counts.get(doc_type, 0) + 1

        confidence = chunk.get("section_confidence")
        if confidence == "inherited":
            inherited_counts[doc_type] = inherited_counts.get(doc_type, 0) + 1
        elif confidence is None:
            no_section_counts[doc_type] = no_section_counts.get(doc_type, 0) + 1

    print("\nDossier Chunking Summary")
    print("------------------------")
    print(f"Total chunks: {len(chunks)}")
    print()

    for doc_type, count in counts.items():
        inherited = inherited_counts.get(doc_type, 0)
        no_section = no_section_counts.get(doc_type, 0)
        print(
            f"{doc_type:35} | {count:4} chunks | "
            f"{inherited:4} inherited section | {no_section:4} no section"
        )


def main() -> None:
    page_records = load_jsonl(INPUT_PATH)
    chunks = chunk_dossier_pages(page_records)
    save_jsonl(chunks, OUTPUT_PATH)
    print_summary(chunks)
    print(f"\nSaved dossier chunks to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()