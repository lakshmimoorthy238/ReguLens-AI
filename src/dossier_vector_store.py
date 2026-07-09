import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "references" / "parsed" / "dossier_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "references" / "indexes"

FAISS_INDEX_PATH = INDEX_DIR / "dossier_faiss.index"
METADATA_PATH = INDEX_DIR / "dossier_metadata.json"
MANIFEST_PATH = INDEX_DIR / "dossier_index_manifest.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(path: Path) -> list[dict]:
    """Load dossier chunks from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

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


def prepare_texts(chunks: list[dict]) -> list[str]:
    """
    Prepare dossier chunk text for embedding.

    Metadata is included in the embedded text to improve retrieval by:
    - file name
    - document type
    - CTD module
    - detected section
    """
    texts = []

    for chunk in chunks:
        dossier_id = chunk.get("dossier_id", "")
        file_name = chunk.get("file_name", "")
        document_type = chunk.get("document_type", "")
        module_guess = chunk.get("module_guess", "")
        section = chunk.get("detected_section", "")
        text = chunk.get("text", "")

        embedding_text = (
            f"Dossier: {dossier_id}\n"
            f"File: {file_name}\n"
            f"Document Type: {document_type}\n"
            f"Module: {module_guess}\n"
            f"Section: {section}\n\n"
            f"{text}"
        )

        texts.append(embedding_text)

    return texts


def build_embeddings(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL_NAME,
) -> np.ndarray:
    """Create normalized embeddings for dossier chunks."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(texts)} dossier chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index.

    Since embeddings are normalized, inner product behaves like cosine similarity.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D numpy array")

    dimension = embeddings.shape[1]

    print(f"Building dossier FAISS index with dimension: {dimension}")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if index.ntotal != embeddings.shape[0]:
        raise RuntimeError(
            f"FAISS index vector count mismatch: "
            f"{index.ntotal} != {embeddings.shape[0]}"
        )

    return index


def build_metadata(chunks: list[dict]) -> list[dict]:
    """
    Store metadata in the same order as FAISS vectors.

    FAISS returns vector IDs, and this metadata maps each vector ID back to:
    - file
    - page
    - chunk
    - source hash
    - evidence text
    """
    metadata = []

    for i, chunk in enumerate(chunks):
        metadata.append(
            {
                "vector_id": i,
                "chunk_id": chunk.get("chunk_id", ""),
                "dossier_id": chunk.get("dossier_id", ""),
                "file_name": chunk.get("file_name", ""),
                "source_hash": chunk.get("source_hash", ""),
                "document_type": chunk.get("document_type", ""),
                "module_guess": chunk.get("module_guess", ""),
                "page_number": chunk.get("page_number"),
                "chunk_number": chunk.get("chunk_number"),
                "detected_section": chunk.get("detected_section"),
                "fresh_section": chunk.get("fresh_section"),
                "section_confidence": chunk.get("section_confidence"),
                "char_count": chunk.get("char_count"),
                "text": chunk.get("text", ""),
            }
        )

    return metadata


def build_manifest(
    embeddings: np.ndarray,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> dict:
    """Record index build metadata."""
    return {
        "index_type": "dossier",
        "embedding_model": model_name,
        "dimension": int(embeddings.shape[1]),
        "num_vectors": int(embeddings.shape[0]),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_chunks": str(CHUNKS_PATH),
    }


def save_index(index: faiss.Index, path: Path) -> None:
    """Save FAISS index to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def save_metadata(metadata: list[dict], path: Path) -> None:
    """Save metadata to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_manifest(manifest: dict, path: Path) -> None:
    """Save index manifest to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def print_summary(chunks: list[dict], metadata: list[dict], manifest: dict) -> None:
    """Print dossier vector store summary."""
    document_counts = {}

    for item in metadata:
        doc_type = item.get("document_type", "unknown")
        document_counts[doc_type] = document_counts.get(doc_type, 0) + 1

    print("\nDossier Vector Store Summary")
    print("----------------------------")
    print(f"Total chunks indexed : {len(chunks)}")
    print(f"Metadata records     : {len(metadata)}")
    print(f"Embedding model      : {manifest['embedding_model']}")
    print(f"Embedding dimension  : {manifest['dimension']}")
    print(f"FAISS index path     : {FAISS_INDEX_PATH}")
    print(f"Metadata path        : {METADATA_PATH}")
    print(f"Manifest path        : {MANIFEST_PATH}")
    print()

    for doc_type, count in document_counts.items():
        print(f"{doc_type:35} | {count:4} vectors")


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)

    if not chunks:
        raise ValueError("No dossier chunks found. Run src/dossier_chunker.py first.")

    texts = prepare_texts(chunks)
    embeddings = build_embeddings(texts)
    index = build_faiss_index(embeddings)
    metadata = build_metadata(chunks)
    manifest = build_manifest(embeddings)

    save_index(index, FAISS_INDEX_PATH)
    save_metadata(metadata, METADATA_PATH)
    save_manifest(manifest, MANIFEST_PATH)

    print_summary(chunks, metadata, manifest)


if __name__ == "__main__":
    main()