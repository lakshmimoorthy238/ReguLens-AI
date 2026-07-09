import json
import hashlib
from pathlib import Path
import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "references" / "guideline_registry.json"
RAW_DIR = PROJECT_ROOT / "references" / "raw"


def load_registry(registry_path: Path = REGISTRY_PATH) -> dict:
    """Load guideline registry JSON."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict, registry_path: Path = REGISTRY_PATH) -> None:
    """Save updated guideline registry JSON."""
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash for a file, streamed in chunks."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def count_pdf_pages(file_path: Path) -> int:
    """Count pages in a PDF using PyMuPDF."""
    with fitz.open(file_path) as doc:
        return doc.page_count


def validate_registry(registry: dict, raw_dir: Path = RAW_DIR) -> dict:
    """
    Lightweight check: does each registry entry have a source_file name,
    and does that file actually exist in references/raw/?

    Does NOT compute hashes or page counts — this is meant to be a fast
    sanity check you can run on its own before doing heavier work.

    Returns a dict of {guideline_id: issue_string} for anything invalid.
    Entries with no issues are omitted.
    """
    issues = {}
    for guideline_id, meta in registry.items():
        source_file = meta.get("source_file", "")
        if not source_file:
            issues[guideline_id] = "missing_source_file_name"
            continue
        pdf_path = raw_dir / source_file
        if not pdf_path.exists():
            issues[guideline_id] = "missing_file"
    return issues


def update_registry_metadata(registry: dict) -> dict:
    """
    Validate guideline files and update metadata:
    - source_hash
    - page_count
    - status

    Preserves "planned" status (fills in hash/page_count but does not
    promote planned guidelines to "registered" — that's a deliberate
    activation step, not automatic).
    """
    issues = validate_registry(registry)

    for guideline_id, meta in registry.items():
        if guideline_id in issues:
            meta["status"] = issues[guideline_id]
            meta["source_hash"] = ""
            meta["page_count"] = None
            continue

        pdf_path = RAW_DIR / meta["source_file"]
        try:
            meta["source_hash"] = calculate_sha256(pdf_path)
            meta["page_count"] = count_pdf_pages(pdf_path)

            current_status = meta.get("status")
            # Don't downgrade an already-indexed guideline, and don't
            # auto-promote a "planned" (not-yet-activated) guideline.
            if current_status not in ("indexed", "planned"):
                meta["status"] = "registered"
        except Exception as exc:
            meta["status"] = "read_error"
            meta["read_error"] = str(exc)
            meta["source_hash"] = ""
            meta["page_count"] = None

    return registry


def print_registry_summary(registry: dict) -> None:
    """Print a clean summary of registry status."""
    total = len(registry)
    registered = sum(1 for m in registry.values() if m.get("status") == "registered")
    indexed = sum(1 for m in registry.values() if m.get("status") == "indexed")
    planned = sum(1 for m in registry.values() if m.get("status") == "planned")
    missing = sum(1 for m in registry.values() if m.get("status") == "missing_file")
    errors = sum(1 for m in registry.values() if m.get("status") == "read_error")
    bad_name = sum(
        1 for m in registry.values() if m.get("status") == "missing_source_file_name"
    )

    print("\nGuideline Registry Summary")
    print("--------------------------")
    print(f"Total guidelines      : {total}")
    print(f"Registered (active)   : {registered}")
    print(f"Indexed                : {indexed}")
    print(f"Planned (not active)  : {planned}")
    print(f"Missing files          : {missing}")
    print(f"Read errors             : {errors}")
    print(f"Missing filename config : {bad_name}")
    print()

    for guideline_id, meta in registry.items():
        version = meta.get("version", "") or ""
        pages = meta.get("page_count", "")
        status = meta.get("status", "")
        file_name = meta.get("source_file", "")
        print(
            f"{guideline_id:12} | "
            f"{version:8} | "
            f"{str(pages):>4} pages | "
            f"{status:24} | "
            f"{file_name}"
        )


def main() -> None:
    registry = load_registry()
    registry = update_registry_metadata(registry)
    save_registry(registry)
    print_registry_summary(registry)


if __name__ == "__main__":
    main()