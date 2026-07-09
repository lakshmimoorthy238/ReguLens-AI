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
FAISS_INDEX_PATH = INDEX_DIR / "guideline_faiss.index"
METADATA_PATH = INDEX_DIR / "guideline_metadata.json"
MANIFEST_PATH = INDEX_DIR / "guideline_index_manifest.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5
DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30

DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35
SECTION_EXACT_MATCH_BOOST = 0.10

# Results scoring below this are treated as noise and dropped before the
# top-k cutoff — this is what actually enforces the "confidence filter"
# from the architecture, rather than just labeling weak results and
# passing them through anyway.
MIN_EVIDENCE_SCORE = 0.35


# Matches CTD-style numeric sections like 3.2.P.8.2, 2.7.4
SECTION_NUMERIC_PATTERN = re.compile(
    r"\b\d+(?:\.(?:\d+|[A-Z]))+\b",
    flags=re.IGNORECASE,
)
# Matches ICH guideline-ID style sections like Q1A, Q2(R2), M4Q — kept in
# sync with chunker.py's detect_section() so a chunk tagged detected_section
# = "Q1A(R2)" can actually be boosted when a query mentions it.
SECTION_GUIDELINE_ID_PATTERN = re.compile(
    r"\b[QEMS]\d+[A-Z]?(?:\(R\d+\))?\b",
    flags=re.IGNORECASE,
)


def load_json(path: Path):
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_faiss_index(path: Path) -> faiss.Index:
    """Load FAISS index from disk."""
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")

    return faiss.read_index(str(path))


def validate_index(index: faiss.Index, metadata: list[dict], manifest: dict) -> None:
    """Validate index, metadata, and manifest compatibility."""
    expected_model = manifest.get("embedding_model")
    expected_dimension = manifest.get("dimension")
    expected_vectors = manifest.get("num_vectors")

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
    """Load embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 keyword search.

    Keeps regulatory-looking tokens such as:
    - 3.2.P.8.2
    - Q1A(R2)
    - M4Q
    """
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:[.\-_/()][a-z0-9]+)*", text)


def build_bm25(metadata: list[dict]) -> BM25Okapi:
    """Build BM25 index from metadata texts."""
    corpus_tokens = []

    for item in metadata:
        searchable_text = make_searchable_text(item)
        corpus_tokens.append(tokenize(searchable_text))

    return BM25Okapi(corpus_tokens)


def make_searchable_text(item: dict) -> str:
    """Create searchable text from metadata record."""
    return (
        f"{item.get('guideline_id', '')} "
        f"{item.get('title', '')} "
        f"{item.get('domain', '')} "
        f"{item.get('detected_section', '')} "
        f"{item.get('text', '')}"
    )


def prepare_query_text(query: str) -> str:
    """Prepare query text for dense embedding."""
    return query.strip()


def dense_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    top_k: int = DENSE_CANDIDATES,
) -> dict[int, float]:
    """Run dense FAISS search and return vector_id -> score."""
    query_text = prepare_query_text(query)

    query_embedding = model.encode(
        [query_text],
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
    """Run BM25 keyword search and return vector_id -> score."""
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    if len(scores) == 0:
        return {}

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = {}

    for idx in top_indices:
        score = float(scores[idx])
        if score > 0:
            results[int(idx)] = score

    return results


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalize scores to 0-1."""
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
    """Extract CTD/ICH-style section or guideline-ID terms from a query."""
    numeric_terms = [m.group(0).lower() for m in SECTION_NUMERIC_PATTERN.finditer(query)]
    guideline_terms = [m.group(0).lower() for m in SECTION_GUIDELINE_ID_PATTERN.finditer(query)]
    return numeric_terms + guideline_terms


def section_boost(query: str, item: dict) -> float:
    """Give small boost when query section exactly appears in metadata/text."""
    section_terms = extract_section_terms(query)

    if not section_terms:
        return 0.0

    detected_section = str(item.get("detected_section", "")).lower()
    text = str(item.get("text", "")).lower()

    for section in section_terms:
        if section == detected_section or section in text:
            return SECTION_EXACT_MATCH_BOOST

    return 0.0


def passes_filters(
    item: dict,
    guideline_id: Optional[str] = None,
    domain_contains: Optional[str] = None,
) -> bool:
    """Apply optional metadata filters."""
    if guideline_id and item.get("guideline_id") != guideline_id:
        return False

    if domain_contains:
        domain = str(item.get("domain", "")).lower()
        if domain_contains.lower() not in domain:
            return False

    return True


def evidence_status(score: float) -> str:
    """Classify retrieval confidence."""
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
    guideline_id: Optional[str] = None,
    domain_contains: Optional[str] = None,
    min_score: float = MIN_EVIDENCE_SCORE,
) -> list[dict]:
    """
    Search guideline index using hybrid retrieval:
    dense FAISS + BM25 keyword search.

    Results scoring below `min_score` are dropped before the top_k cutoff —
    this is the actual "confidence filter" enforcement. Pass min_score=0.0
    to disable filtering (e.g. for debugging what the raw candidate pool
    looks like).
    """
    dense_scores = dense_search(query, model, index)
    bm25_scores = bm25_search(query, bm25)

    dense_norm = normalize_scores(dense_scores)
    bm25_norm = normalize_scores(bm25_scores)

    candidate_ids = set(dense_scores.keys()) | set(bm25_scores.keys())

    results = []

    for vector_id in candidate_ids:
        item = metadata[vector_id]

        if not passes_filters(item, guideline_id, domain_contains):
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
            "guideline_id": item.get("guideline_id", ""),
            "title": item.get("title", ""),
            "version": item.get("version", ""),
            "domain": item.get("domain", ""),
            "source_file": item.get("source_file", ""),
            "source_hash": item.get("source_hash", ""),
            "page_number": item.get("page_number"),
            "detected_section": item.get("detected_section"),
            "section_confidence": item.get("section_confidence"),
            "text": item.get("text", ""),
        }

        results.append(result)

    results = sorted(results, key=lambda r: r["retrieval_score"], reverse=True)

    return results[:top_k]


def print_results(query: str, results: list[dict]) -> None:
    """Print retrieval results in readable form."""
    print("\nQuery")
    print("-----")
    print(query)

    print("\nTop Retrieved Guideline Chunks")
    print("------------------------------")

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
        print(f"Guideline      : {result['guideline_id']} {result['version']}")
        print(f"Domain         : {result['domain']}")
        print(f"Section        : {result['detected_section']} ({result['section_confidence']})")
        print(f"Source         : {result['source_file']} | page {result['page_number']}")
        print(f"Chunk ID       : {result['chunk_id']}")
        print(f"Snippet        : {snippet}")


def run_default_tests(
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    bm25: BM25Okapi,
) -> None:
    """Run a few default test queries."""
    test_queries = [
        "What is expected in 3.2.P.8.2 post approval stability protocol?",
        "Clinical study report synopsis requirements",
        "stability testing shelf life storage condition",
        "validation of analytical procedures specificity accuracy precision",
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
    parser = argparse.ArgumentParser(description="Guideline RAG retriever")
    parser.add_argument("--query", type=str, default=None, help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results")
    parser.add_argument("--guideline", type=str, default=None, help="Optional guideline_id filter")
    parser.add_argument("--domain", type=str, default=None, help="Optional domain substring filter")
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_EVIDENCE_SCORE,
        help="Minimum retrieval score to keep a result (set 0 to disable filtering)",
    )

    args = parser.parse_args()

    manifest = load_json(MANIFEST_PATH)
    metadata = load_json(METADATA_PATH)
    index = load_faiss_index(FAISS_INDEX_PATH)

    validate_index(index, metadata, manifest)

    model = load_embedding_model()

    print("Building BM25 keyword index from metadata...")
    bm25 = build_bm25(metadata)

    if args.query:
        results = hybrid_search(
            query=args.query,
            model=model,
            index=index,
            metadata=metadata,
            bm25=bm25,
            top_k=args.top_k,
            guideline_id=args.guideline,
            domain_contains=args.domain,
            min_score=args.min_score,
        )
        print_results(args.query, results)
    else:
        run_default_tests(model, index, metadata, bm25)


if __name__ == "__main__":
    main()