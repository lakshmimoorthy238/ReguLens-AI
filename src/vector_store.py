import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "references" / "parsed" / "guideline_chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "references" / "indexes"

FAISS_INDEX_PATH = INDEX_DIR / "guideline_faiss.index"
METADATA_PATH = INDEX_DIR / "guideline_metadata.json"
MANIFEST_PATH = INDEX_DIR / "guideline_index_manifest.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(path: Path) -> list[dict]:
    """Load guideline chunks from JSONL."""
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
    Prepare text for embedding.

    We include some metadata (guideline, title, domain, section) in the
    embedded text to help cluster chunks by topic. Page number is
    deliberately excluded — it carries no semantic meaning and only adds
    noise to the embedding. Note: this prefix helps topical grouping but
    is NOT a substitute for exact section-number matching, since dense
    embeddings are weak at literal token/number matching. Exact-match
    retrieval (e.g. via BM25) is a planned upgrade, not covered here.
    """
    texts = []

    for chunk in chunks:
        guideline_id = chunk.get("guideline_id", "")
        title = chunk.get("title", "")
        domain = chunk.get("domain", "")
        section = chunk.get("detected_section", "")
        text = chunk.get("text", "")

        embedding_text = (
            f"Guideline: {guideline_id}\n"
            f"Title: {title}\n"
            f"Domain: {domain}\n"
            f"Section: {section}\n\n"
            f"{text}"
        )

        texts.append(embedding_text)

    return texts


def build_embeddings(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Create normalized embeddings for all chunks."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(texts)} chunks...")
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

    Since embeddings are normalized, inner product works like cosine similarity.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D numpy array")

    dimension = embeddings.shape[1]

    print(f"Building FAISS index with dimension: {dimension}")

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

    FAISS returns vector indices. We use those indices to recover metadata.
    """
    metadata = []

    for i, chunk in enumerate(chunks):
        metadata.append(
            {
                "vector_id": i,
                "chunk_id": chunk.get("chunk_id", ""),
                "guideline_id": chunk.get("guideline_id", ""),
                "title": chunk.get("title", ""),
                "version": chunk.get("version", ""),
                "domain": chunk.get("domain", ""),
                "source_file": chunk.get("source_file", ""),
                "source_hash": chunk.get("source_hash", ""),
                "index_version": chunk.get("index_version", ""),
                "page_number": chunk.get("page_number"),
                "chunk_number": chunk.get("chunk_number"),
                "detected_section": chunk.get("detected_section"),
                "section_confidence": chunk.get("section_confidence"),
                "char_count": chunk.get("char_count"),
                "text": chunk.get("text", ""),
            }
        )

    return metadata


def build_manifest(embeddings: np.ndarray, model_name: str = EMBEDDING_MODEL_NAME) -> dict:
    """
    Record which embedding model and dimension built this index.

    retriever.py should check this before querying, so swapping embedding
    models later can't silently load a mismatched index.
    """
    return {
        "embedding_model": model_name,
        "dimension": int(embeddings.shape[1]),
        "num_vectors": int(embeddings.shape[0]),
        "built_at": datetime.now(timezone.utc).isoformat(),
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
    """Print vector store summary."""
    guideline_counts = {}

    for item in metadata:
        guideline_id = item.get("guideline_id", "UNKNOWN")
        guideline_counts[guideline_id] = guideline_counts.get(guideline_id, 0) + 1

    print("\nGuideline Vector Store Summary")
    print("------------------------------")
    print(f"Total chunks indexed : {len(chunks)}")
    print(f"Metadata records     : {len(metadata)}")
    print(f"Embedding model      : {manifest['embedding_model']}")
    print(f"Embedding dimension  : {manifest['dimension']}")
    print(f"FAISS index path     : {FAISS_INDEX_PATH}")
    print(f"Metadata path        : {METADATA_PATH}")
    print(f"Manifest path        : {MANIFEST_PATH}")
    print()

    for guideline_id, count in guideline_counts.items():
        print(f"{guideline_id:12} | {count:4} vectors")


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)

    if not chunks:
        raise ValueError("No chunks found. Run src/chunker.py first.")

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