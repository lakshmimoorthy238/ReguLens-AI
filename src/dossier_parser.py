import json
import re
from pathlib import Path

import fitz  # PyMuPDF

import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOSSIER_ID = "sample_dossier_v1"
DOSSIER_DIR = PROJECT_ROOT / "sample_dossier"

OUTPUT_DIR = PROJECT_ROOT / "references"/"parsed"
OUTPUT_PATH = OUTPUT_DIR / "dossier_pages.jsonl"

LOW_TEXT_CHAR_THRESHOLD = 20

def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash for a file, streamed in chunks."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()

def clean_text(text: str) -> str:
    """Basic PDF text cleanup."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_document_type(file_name: str) -> str:
    """
    Infer document type from filename.

    This is simple filename-based detection for MVP.
    Later, your fine-tuned LLM can classify sections more intelligently.
    """
    name = file_name.lower()

    if "qos" in name or "quality_overall" in name:
        return "module_2_qos"

    if "clinical_summary" in name or "module_2_clinical" in name:
        return "module_2_clinical_summary"

    if "stability" in name:
        return "module_3_stability"

    if "manufacturing" in name or "manufacturer" in name:
        return "module_3_manufacturing"

    if "nonclinical" in name or "module_4" in name:
        return "module_4_nonclinical"

    if "clinical_study" in name or "csr" in name or "module_5" in name:
        return "module_5_clinical_study_report"

    if "label" in name or "labelling" in name or "labeling" in name:
        return "label"

    return "unknown"


def infer_module_guess(document_type: str) -> str:
    """Infer CTD module from document type."""
    if document_type.startswith("module_2"):
        return "Module 2"
    if document_type.startswith("module_3"):
        return "Module 3"
    if document_type.startswith("module_4"):
        return "Module 4"
    if document_type.startswith("module_5"):
        return "Module 5"
    if document_type == "label":
        return "Label"
    return "Unknown"


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract page-wise text from a dossier PDF."""
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


def list_dossier_pdfs(dossier_dir: Path = DOSSIER_DIR) -> list[Path]:
    """List PDF files in sample_dossier folder."""
    if not dossier_dir.exists():
        raise FileNotFoundError(f"Dossier folder not found: {dossier_dir}")

    pdf_files = sorted(dossier_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {dossier_dir}. "
            f"Add your sample dossier PDFs before running this parser."
        )

    return pdf_files


def parse_dossier_pdfs() -> tuple[list[dict], list[dict]]:
    """
    Parse all PDFs in sample_dossier into page-level records.

    Returns:
        records: extracted page records
        parse_failures: failed PDFs and errors
    """
    all_records = []
    parse_failures = []

    pdf_files = list_dossier_pdfs()

    for pdf_path in pdf_files:
        file_name = pdf_path.name
        document_type = infer_document_type(file_name)
        module_guess = infer_module_guess(document_type)

        if document_type == "unknown":
            print(
                f"  WARNING: could not classify '{file_name}' from its filename "
                f"(document_type=unknown). It will still be parsed and included, "
                f"but check infer_document_type() patterns or rename the file."
            )

        print(f"Parsing {file_name} as {document_type}")

        try:
            source_hash = calculate_sha256(pdf_path)
        except Exception as exc:
            print(f"  WARNING: could not hash {file_name}: {exc}")
            source_hash = ""

        try:
            pages = extract_pdf_pages(pdf_path)
        except Exception as exc:
            print(f"  FAILED to parse {file_name}: {exc}")
            parse_failures.append(
                {
                    "file_name": file_name,
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
                f"page(s): {preview}{extra}"
            )

        for page in pages:
            record = {
                "dossier_id": DOSSIER_ID,
                "file_name": file_name,
                "source_hash": source_hash,
                "document_type": document_type,
                "module_guess": module_guess,
                "page_number": page["page_number"],
                "text": page["text"],
                "char_count": page["char_count"],
                "possibly_scanned": page["possibly_scanned"],
            }

            all_records.append(record)

    return all_records, parse_failures


def save_jsonl(records: list[dict], output_path: Path = OUTPUT_PATH) -> None:
    """Save records as JSONL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(records: list[dict], parse_failures: list[dict]) -> None:
    """Print dossier parsing summary."""
    document_counts = {}
    scanned_counts = {}

    for record in records:
        doc_type = record["document_type"]
        document_counts[doc_type] = document_counts.get(doc_type, 0) + 1

        if record["possibly_scanned"]:
            scanned_counts[doc_type] = scanned_counts.get(doc_type, 0) + 1

    print("\nDossier PDF Parsing Summary")
    print("---------------------------")
    print(f"Dossier ID         : {DOSSIER_ID}")
    print(f"Total page records : {len(records)}")
    print(f"Parse failures     : {len(parse_failures)}")
    print()

    for doc_type, count in document_counts.items():
        scanned = scanned_counts.get(doc_type, 0)
        flag = f"  ({scanned} possibly scanned/empty)" if scanned else ""
        warn = "  <-- check filename" if doc_type == "unknown" else ""
        print(f"{doc_type:35} | {count:4} pages{flag}{warn}")

    if parse_failures:
        print("\nFailed PDFs:")
        for failure in parse_failures:
            print(f"  {failure['file_name']}: {failure['error']}")


def main() -> None:
    records, parse_failures = parse_dossier_pdfs()
    save_jsonl(records)
    print_summary(records, parse_failures)
    print(f"\nSaved dossier pages to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()