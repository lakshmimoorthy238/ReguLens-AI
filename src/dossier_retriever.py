import argparse
import json
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "references" / "indexes"

FAISS_INDEX_PATH = INDEX_DIR / "dossier_faiss.index"
METADATA_PATH = INDEX_DIR / "dossier_metadata.json"
MANIFEST_PATH = INDEX_DIR / "dossier_index_manifest.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_INDEX_TYPE = "dossier"

DEFAULT_TOP_K = 5
DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30

DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35
SECTION_EXACT_MATCH_BOOST = 0.10
MIN_EVIDENCE_SCORE = 0.35

# Matches CTD-style dotted section codes: 3.2.P.8.2, 2.7.4, etc.
CTD_SECTION_PATTERN = re.compile(
    r"\b\d+(?:\.(?:\d+|[A-Z]))+\b",
    flags=re.IGNORECASE,
)
# Matches simple top-level numbered headings used in ICH E3-style CSR
# documents (e.g. "7. Efficacy Evaluation") — kept in sync with the same
# fallback pattern in dossier_chunker.py so boosting actually works for
# chunks whose detected_section came from that fallback.
SIMPLE_SECTION_PATTERN = re.compile(
    r"\bsection\s+(\d{1,2})\b|\b(\d{1,2})\.\s+[A-Z][a-z]+",
    flags=re.IGNORECASE,
)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_faiss_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")

    return faiss.read_index(str(path))


def validate_index(index: faiss.Index, metadata: list[dict], manifest: dict) -> None:
    expected_model = manifest.get("embedding_model")
    expected_dimension = manifest.get("dimension")
    expected_vectors = manifest.get("num_vectors")
    index_type = manifest.get("index_type")

    # Guards against accidentally loading the guideline index here — both
    # indexes share the same embedding model and dimension, so none of the
    # other checks below would catch that particular mistake on their own.
    if index_type is not None and index_type != EXPECTED_INDEX_TYPE:
        raise ValueError(
            f"Index type mismatch. "
            f"Manifest says index_type={index_type!r}, "
            f"but this is the {EXPECTED_INDEX_TYPE} retriever."
        )

    if expected_model != EMBEDDING_MODEL_NAME:
        raise ValueError(
            f"Embedding model mismatch. "
            f"Index was built with {expected_model}, "
            f"but retriever expects {EMBEDDING_MODEL_NAME}."
        )

    if index.d != expected_dimension:
        raise ValueError(
            f"FAISS dimension mismatch. "
            f"Index dimension={index.d}, manifest dimension={expected_dimension}."
        )

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata count mismatch. "
            f"FAISS vectors={index.ntotal}, metadata records={len(metadata)}."
        )

    if expected_vectors != len(metadata):
        raise ValueError(
            f"Manifest/metadata count mismatch. "
            f"Manifest vectors={expected_vectors}, metadata records={len(metadata)}."
        )


def load_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:[.\-_/()][a-z0-9]+)*", text)


def make_searchable_text(item: dict) -> str:
    return (
        f"{item.get('dossier_id', '')} "
        f"{item.get('file_name', '')} "
        f"{item.get('document_type', '')} "
        f"{item.get('module_guess', '')} "
        f"{item.get('detected_section', '')} "
        f"{item.get('text', '')}"
    )


def build_bm25(metadata: list[dict]) -> BM25Okapi:
    corpus_tokens = []

    for item in metadata:
        corpus_tokens.append(tokenize(make_searchable_text(item)))

    return BM25Okapi(corpus_tokens)


def dense_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    top_k: int = DENSE_CANDIDATES,
) -> dict[int, float]:
    query_embedding = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = {}

    for vector_id, score in zip(indices[0], scores[0]):
        if vector_id == -1:
            continue

        results[int(vector_id)] = float(score)

    return results


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    top_k: int = BM25_CANDIDATES,
) -> dict[int, float]:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    if len(scores) == 0:
        return {}

    top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = {}

    for idx in top_indices:
        score = float(scores[idx])
        if score > 0:
            results[int(idx)] = score

    return results


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        return {idx: 1.0 for idx in scores}

    return {
        idx: (score - min_score) / (max_score - min_score)
        for idx, score in scores.items()
    }


def extract_section_terms(query: str) -> list[str]:
    """Extract CTD-dotted terms and simple 'section N' style references."""
    ctd_terms = [m.group(0).lower() for m in CTD_SECTION_PATTERN.finditer(query)]

    simple_terms = []
    for m in SIMPLE_SECTION_PATTERN.finditer(query):
        number = m.group(1) or m.group(2)
        if number:
            simple_terms.append(number)

    return ctd_terms + simple_terms


def section_boost(query: str, item: dict) -> float:
    section_terms = extract_section_terms(query)

    if not section_terms:
        return 0.0

    detected_section = str(item.get("detected_section", "")).lower()
    text = str(item.get("text", "")).lower()

    for section in section_terms:
        if section == detected_section or section in text:
            return SECTION_EXACT_MATCH_BOOST
        # Handles simple headings like detected_section = "7. efficacy
        # evaluation" when the query just says "section 7" or "7".
        if detected_section.startswith(f"{section}."):
            return SECTION_EXACT_MATCH_BOOST

    return 0.0


def passes_filters(
    item: dict,
    document_type: Optional[str] = None,
    file_contains: Optional[str] = None,
    module_contains: Optional[str] = None,
) -> bool:
    if document_type and item.get("document_type") != document_type:
        return False

    if file_contains:
        file_name = str(item.get("file_name", "")).lower()
        if file_contains.lower() not in file_name:
            return False

    if module_contains:
        module_guess = str(item.get("module_guess", "")).lower()
        if module_contains.lower() not in module_guess:
            return False

    return True


def evidence_status(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "medium"
    return "weak"


def hybrid_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    bm25: BM25Okapi,
    top_k: int = DEFAULT_TOP_K,
    document_type: Optional[str] = None,
    file_contains: Optional[str] = None,
    module_contains: Optional[str] = None,
    min_score: float = MIN_EVIDENCE_SCORE,
) -> list[dict]:
    dense_scores = dense_search(query, model, index)
    bm25_scores = bm25_search(query, bm25)

    dense_norm = normalize_scores(dense_scores)
    bm25_norm = normalize_scores(bm25_scores)

    candidate_ids = set(dense_scores.keys()) | set(bm25_scores.keys())

    results = []

    for vector_id in candidate_ids:
        item = metadata[vector_id]

        if not passes_filters(
            item=item,
            document_type=document_type,
            file_contains=file_contains,
            module_contains=module_contains,
        ):
            continue

        dense_component = dense_norm.get(vector_id, 0.0)
        bm25_component = bm25_norm.get(vector_id, 0.0)
        boost = section_boost(query, item)

        final_score = (
            DENSE_WEIGHT * dense_component
            + BM25_WEIGHT * bm25_component
            + boost
        )

        if final_score < min_score:
            continue

        result = {
            "retrieval_score": round(float(final_score), 4),
            "evidence_status": evidence_status(float(final_score)),
            "dense_score": round(float(dense_scores.get(vector_id, 0.0)), 4),
            "bm25_score": round(float(bm25_scores.get(vector_id, 0.0)), 4),
            "vector_id": vector_id,
            "chunk_id": item.get("chunk_id", ""),
            "dossier_id": item.get("dossier_id", ""),
            "file_name": item.get("file_name", ""),
            "source_hash": item.get("source_hash", ""),
            "document_type": item.get("document_type", ""),
            "module_guess": item.get("module_guess", ""),
            "page_number": item.get("page_number"),
            "chunk_number": item.get("chunk_number"),
            "detected_section": item.get("detected_section"),
            "section_confidence": item.get("section_confidence"),
            "citation": {
                "file_name": item.get("file_name", ""),
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id", ""),
                "source_hash": item.get("source_hash", ""),
            },
            "text": item.get("text", ""),
        }

        results.append(result)

    results = sorted(results, key=lambda r: r["retrieval_score"], reverse=True)

    return results[:top_k]


def print_results(query: str, results: list[dict]) -> None:
    print("\nQuery")
    print("-----")
    print(query)

    print("\nTop Retrieved Dossier Chunks")
    print("----------------------------")

    if not results:
        print("No results found above the confidence threshold.")
        return

    for rank, result in enumerate(results, start=1):
        text = result.get("text", "")
        snippet = re.sub(r"\s+", " ", text).strip()[:700]

        print(f"\nResult {rank}")
        print(f"Score          : {result['retrieval_score']} ({result['evidence_status']})")
        print(f"Dense score    : {result['dense_score']}")
        print(f"BM25 score     : {result['bm25_score']}")
        print(f"File           : {result['file_name']}")
        print(f"Document type  : {result['document_type']}")
        print(f"Module         : {result['module_guess']}")
        print(f"Section        : {result['detected_section']} ({result['section_confidence']})")
        print(f"Page           : {result['page_number']}")
        print(f"Chunk ID       : {result['chunk_id']}")
        print(f"Snippet        : {snippet}")


def run_default_tests(
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    bm25: BM25Okapi,
) -> None:
    test_queries = [
        "proposed shelf life 24 months store below 25",
        "shelf life 18 months store below 30 label",
        "primary endpoint systolic blood pressure reduction Week 12",
        "primary endpoint heart rate reduction Week 12",
        "serious adverse events 3 serious adverse events 5",
        "manufacturer Apex Pharma Nova Labs",
        "batch B003 B001 B002",
        "clinical study report synopsis",
        "rapid control of blood pressure within 24 hours",
        "section 7 efficacy evaluation",
    ]

    for query in test_queries:
        results = hybrid_search(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            bm25=bm25,
            top_k=5,
        )
        print_results(query, results)
        print("\n" + "=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dossier RAG retriever")

    parser.add_argument("--query", type=str, default=None, help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results")
    parser.add_argument("--document-type", type=str, default=None, help="Optional document_type filter")
    parser.add_argument("--file", type=str, default=None, help="Optional filename substring filter")
    parser.add_argument("--module", type=str, default=None, help="Optional module substring filter")
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_EVIDENCE_SCORE,
        help="Minimum retrieval score to keep a result. Use 0 for debugging.",
    )

    args = parser.parse_args()

    manifest = load_json(MANIFEST_PATH)
    metadata = load_json(METADATA_PATH)
    index = load_faiss_index(FAISS_INDEX_PATH)

    validate_index(index, metadata, manifest)

    model = load_embedding_model()

    print("Building BM25 keyword index from dossier metadata...")
    bm25 = build_bm25(metadata)

    if args.query:
        results = hybrid_search(
            query=args.query,
            model=model,
            index=index,
            metadata=metadata,
            bm25=bm25,
            top_k=args.top_k,
            document_type=args.document_type,
            file_contains=args.file,
            module_contains=args.module,
            min_score=args.min_score,
        )
        print_results(args.query, results)
    else:
        run_default_tests(model, index, metadata, bm25)


if __name__ == "__main__":
    main()