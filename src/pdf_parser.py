import json
import re
from pathlib import Path

import fitz  # PyMuPDF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "references" / "guideline_registry.json"
RAW_DIR = PROJECT_ROOT / "references" / "raw"
PARSED_DIR = PROJECT_ROOT / "references" / "parsed"
OUTPUT_PATH = PARSED_DIR / "guideline_pages.jsonl"

# Pages with fewer than this many characters are flagged as possibly
# scanned/image-only or empty.
LOW_TEXT_CHAR_THRESHOLD = 20


def load_registry() -> dict:
    """Load guideline registry JSON."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """Basic PDF text cleanup."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Extract page-wise text from a PDF.

    Each page record includes:
    - page_number
    - extracted text
    - character count
    - possibly_scanned flag
    """
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            raw_text = page.get_text("text")
            cleaned_text = clean_text(raw_text)

            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                    "possibly_scanned": len(cleaned_text) < LOW_TEXT_CHAR_THRESHOLD,
                }
            )

    return pages


def parse_registered_guidelines() -> tuple[list[dict], list[dict]]:
    """
    Parse all registered/indexed guideline PDFs into page-level records.

    Guidelines with status other than registered/indexed are skipped.
    One failed PDF will not stop the full parsing run.

    Returns:
        records: page-level guideline records
        parse_failures: list of failed guideline IDs and errors
    """
    registry = load_registry()
    all_records = []
    parse_failures = []

    for guideline_id, meta in registry.items():
        status = meta.get("status")

        if status not in ("registered", "indexed"):
            print(f"Skipping {guideline_id}: status={status}")
            continue

        source_file = meta.get("source_file", "")
        pdf_path = RAW_DIR / source_file

        if not source_file:
            print(f"Skipping {guideline_id}: missing source_file")
            continue

        if not pdf_path.exists():
            print(f"Skipping {guideline_id}: file not found {pdf_path}")
            continue

        print(f"Parsing {guideline_id}: {source_file}")

        try:
            pages = extract_pdf_pages(pdf_path)
        except Exception as exc:
            print(f"  FAILED to parse {guideline_id}: {exc}")
            parse_failures.append(
                {
                    "guideline_id": guideline_id,
                    "source_file": source_file,
                    "error": str(exc),
                }
            )
            continue

        scanned_pages = [p["page_number"] for p in pages if p["possibly_scanned"]]

        if scanned_pages:
            preview = scanned_pages[:10]
            extra = "..." if len(scanned_pages) > 10 else ""
            print(
                f"  Warning: {len(scanned_pages)} possibly scanned/empty "
                f"page(s) in {guideline_id}: {preview}{extra}"
            )

        for page in pages:
            record = {
                "guideline_id": guideline_id,
                "title": meta.get("title", ""),
                "version": meta.get("version", ""),
                "domain": meta.get("domain", ""),
                "source_file": source_file,
                "source_hash": meta.get("source_hash", ""),
                "index_version": meta.get("index_version", ""),
                "page_number": page["page_number"],
                "text": page["text"],
                "char_count": page["char_count"],
                "possibly_scanned": page["possibly_scanned"],
            }
            all_records.append(record)

    return all_records, parse_failures


def save_jsonl(records: list[dict], output_path: Path = OUTPUT_PATH) -> None:
    """Save records as JSONL, one JSON object per line."""
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(records: list[dict], parse_failures: list[dict]) -> None:
    """Print parsing summary."""
    guideline_counts = {}
    scanned_counts = {}

    for record in records:
        guideline_id = record["guideline_id"]
        guideline_counts[guideline_id] = guideline_counts.get(guideline_id, 0) + 1

        if record["possibly_scanned"]:
            scanned_counts[guideline_id] = scanned_counts.get(guideline_id, 0) + 1

    print("\nGuideline PDF Parsing Summary")
    print("-----------------------------")
    print(f"Total page records : {len(records)}")
    print(f"Parse failures     : {len(parse_failures)}")
    print()

    for guideline_id, count in guideline_counts.items():
        scanned = scanned_counts.get(guideline_id, 0)
        flag = f"  ({scanned} possibly scanned/empty)" if scanned else ""
        print(f"{guideline_id:12} | {count:4} pages{flag}")

    if parse_failures:
        print("\nFailed guidelines:")
        for failure in parse_failures:
            print(
                f"  {failure['guideline_id']} "
                f"({failure['source_file']}): {failure['error']}"
            )


def main() -> None:
    records, parse_failures = parse_registered_guidelines()
    save_jsonl(records)
    print_summary(records, parse_failures)
    print(f"\nSaved parsed pages to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()