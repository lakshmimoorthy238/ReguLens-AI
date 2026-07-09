"""
src/rule_verifier.py

Rule-Based Verification Layer for the regulatory dossier assistant.

Purpose:
- Double-check deterministic issues that should not rely only on the LLM.
- Verify mismatches such as shelf life, storage condition, manufacturer,
  batch, primary endpoint, SAE count, missing CTD sections, and CSR synopsis.
- Write evidence-backed rule verification results for reconciliation/reporting.

Run from project root:
    python -m src.rule_verifier

Typical inputs:
    outputs/llm_decisions.json
    outputs/bidirectional_evidence_packages.json
    references/parsed/dossier_chunks.jsonl

Output:
    outputs/rule_verification_results.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_LLM_DECISIONS = Path("outputs/llm_decisions.json")
DEFAULT_EVIDENCE_PACKAGES = Path("outputs/bidirectional_evidence_packages.json")
DEFAULT_DOSSIER_CHUNKS = Path("references/parsed/dossier_chunks.jsonl")
DEFAULT_OUTPUT = Path("outputs/rule_verification_results.json")


# ---------------------------------------------------------------------------
# Deterministic pattern definitions
# ---------------------------------------------------------------------------

SHELF_LIFE_RE = re.compile(
    r"(?i)\b(?:shelf[-\s]*life|proposed\s+shelf[-\s]*life)\b"
    r"[^.\n;]{0,100}?"
    r"[:\s]*"
    r"(?P<value>\d{1,2})\s*months?\b"
)

# Also catches compact forms such as "24-month shelf life".
SHELF_LIFE_PREFIX_RE = re.compile(
    r"(?i)\b(?P<value>\d{1,2})\s*[- ]?month\s+shelf[-\s]*life\b"
)

STORAGE_RE = re.compile(
    r"(?i)\b(?:store|storage condition|storage|stored|keep|do\s+not\s+store)\b"
    r"[^.\n;]{0,100}?"
    r"(?:(?:below|above|at|between|not\s+above|do\s+not\s+store\s+above)\s*)?"
    r"(?P<value>(?:\d{1,2}\s*-\s*\d{1,2})|\d{1,2})\s*°?\s*C\b"
)

# Manufacturer phrases in the synthetic dossier are simple enough to capture.
COMPANY_NAME_RE = re.compile(
    r"\b(?P<value>"
    r"[A-Z][A-Za-z0-9&'\-]*"
    r"(?:\s+[A-Z][A-Za-z0-9&'\-.]*){1,8}"
    r"\s+(?:Pvt\.?\s+Ltd\.?|Private\s+Limited|Ltd\.?|Inc\.?|GmbH|S\.A\.?|PLC)"
    r")\b"
)

KNOWN_MANUFACTURER_RE = re.compile(
    r"(?i)\b(?P<value>"
    r"Apex\s+Pharma\s+Manufacturing\s+Pvt\.?\s+Ltd\.?|"
    r"Nova\s+Labs\s+Manufacturing\s+Pvt\.?\s+Ltd\.?|"
    r"Apex\s+Pharma\s+Manufacturing\s+Private\s+Limited|"
    r"Nova\s+Labs\s+Manufacturing\s+Private\s+Limited"
    r")\b"
)

MANUFACTURER_CONTEXT_RE = re.compile(
    r"\b("
    r"manufacturer|manufacturing|manufactured|manufacture|"
    r"site|facility|qos|quality overall summary|module 3|"
    r"drug product manufacturer|drug substance manufacturer"
    r")\b",
    re.IGNORECASE,
)

PRIMARY_ENDPOINT_SENTENCE_RE = re.compile(
    r"(?i)\bprimary\s+(?:efficacy\s+)?endpoint\b"
    r"[^.\n;]{0,80}"
    r"(?:is|was|as|:|defined\s+as|reported\s+as|specified\s+as)?"
    r"\s*['\"]?"
    r"(?P<value>[^.\n;]{4,160})"
)

# Batch IDs seen in the synthetic data and demo dossier.
BATCH_RE = re.compile(
    r"\b(?:B\d{3}|STB[- ]?\d{3}|VAL[- ]?\d{3}|BN[- ]?\d{4}|LOT[- ]?\d{4}|PB[- ]?\d{4})\b",
    flags=re.IGNORECASE,
)


SAE_RE = re.compile(
    r"(?i)\b(?P<value>\d{1,3})\s+(?:serious\s+adverse\s+events?|SAEs?)\b"
)

SECTION_32P82_RE = re.compile(
    r"(?i)\b3\.2\.P\.8\.2\b|post[-\s]*approval\s+stability\s+protocol|stability\s+commitment"
)

SECTION_32P81_RE = re.compile(r"(?i)\b3\.2\.P\.8\.1\b")
SECTION_32P83_RE = re.compile(r"(?i)\b3\.2\.P\.8\.3\b")
SYNOPSIS_RE = re.compile(r"(?i)\bsynopsis\b")


DETERMINISTIC_FINDING_TYPES = {
    "missing_section",
    "value_inconsistency",
    "entity_inconsistency",
    "endpoint_inconsistency",
    "safety_count_inconsistency",
}


@dataclass
class EvidenceHit:
    value: str
    file_name: Optional[str]
    page_number: Optional[int]
    section: Optional[str]
    chunk_id: Optional[str]
    text_snippet: str


@dataclass
class RuleFinding:
    rule_id: str
    rule_name: str
    rule_status: str
    severity: str
    finding_type: str
    finding_summary: str
    evidence_status: str
    evidence: List[Dict[str, Any]]
    reviewer_action: str


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generic extraction helpers
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\u00a0", " ")


def lower_clean(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).strip().lower())


def canonical_manufacturer(value: str) -> str:
    text = lower_clean(value)
    replacements = {
        "private limited": "pvt ltd",
        "pvt. ltd.": "pvt ltd",
        "pvt ltd.": "pvt ltd",
        "limited": "ltd",
        "ltd.": "ltd",
        "inc.": "inc",
        "co.": "co",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_endpoint(value: str) -> str:
    text = lower_clean(value)
    text = re.sub(r"['\"()]", "", text)
    text = re.sub(r"\s+", " ", text)
    # Remove common trailing filler.
    text = re.sub(r"\s+(for|in)\s+[a-z0-9 \-]+$", "", text)
    return text.strip()


def canonical_storage(value: str) -> str:
    text = lower_clean(value)
    text = text.replace("degrees celsius", "c")
    text = text.replace("degree celsius", "c")
    text = text.replace("\u00b0c", "c")
    text = text.replace(" ", "")
    return text


def make_snippet(text: str, start: int = 0, end: int = 200) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    lo = max(0, start - 90)
    hi = min(len(text), max(end + 90, lo + 180))
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def get_record_text(record: Dict[str, Any]) -> str:
    for key in ("text", "content_text", "requirement_text", "content", "page_text", "chunk_text"):
        if isinstance(record.get(key), str) and record[key].strip():
            return record[key]
    return ""


def get_file_name(record: Dict[str, Any]) -> Optional[str]:
    for key in ("file_name", "source_file", "document", "document_name", "module_ref"):
        value = record.get(key)
        if value:
            return str(value)
    source = record.get("source")
    if isinstance(source, dict):
        return get_file_name(source)
    return None


def get_page_number(record: Dict[str, Any]) -> Optional[int]:
    for key in ("page_number", "page", "page_num"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    source = record.get("source")
    if isinstance(source, dict):
        return get_page_number(source)
    return None


def get_section(record: Dict[str, Any]) -> Optional[str]:
    for key in ("section", "section_ref", "ctd_section", "module_ref"):
        value = record.get(key)
        if value:
            return str(value)
    source = record.get("source")
    if isinstance(source, dict):
        return get_section(source)
    return None


def get_chunk_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("chunk_id", "id"):
        value = record.get(key)
        if value:
            return str(value)
    source = record.get("source")
    if isinstance(source, dict):
        return get_chunk_id(source)
    return None


def iter_text_records(obj: Any, parent_meta: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
    """Recursively yield records that contain text-like fields."""
    parent_meta = parent_meta or {}

    if isinstance(obj, dict):
        merged_meta = dict(parent_meta)
        for key in ("file_name", "source_file", "page_number", "page", "section", "section_ref", "chunk_id", "module_ref"):
            if key in obj and key not in merged_meta:
                merged_meta[key] = obj[key]

        text = get_record_text(obj)
        if text:
            rec = dict(merged_meta)
            rec.update(obj)
            yield rec

        for value in obj.values():
            yield from iter_text_records(value, merged_meta)

    elif isinstance(obj, list):
        for item in obj:
            yield from iter_text_records(item, parent_meta)


def extract_hits(records: Iterable[Dict[str, Any]], pattern: re.Pattern, value_group: str = "value") -> List[EvidenceHit]:
    hits: List[EvidenceHit] = []
    for rec in records:
        text = get_record_text(rec)
        if not text:
            continue
        for match in pattern.finditer(text):
            value = match.groupdict().get(value_group, match.group(0))
            hits.append(EvidenceHit(
                value=str(value).strip(" '\"."),
                file_name=get_file_name(rec),
                page_number=get_page_number(rec),
                section=get_section(rec),
                chunk_id=get_chunk_id(rec),
                text_snippet=make_snippet(text, match.start(), match.end()),
            ))
    return hits

def extract_manufacturer_hits(records: Iterable[Dict[str, Any]]) -> List[EvidenceHit]:
    hits: List[EvidenceHit] = []

    for rec in records:
        text = get_record_text(rec)
        if not text:
            continue

        file_name = get_file_name(rec)
        file_hint = file_name.lower()

        file_has_context = any(
            key in file_hint
            for key in [
                "manufacturing",
                "manufacturer",
                "qos",
                "quality",
                "module_2_qos",
                "module_3_manufacturing",
            ]
        )

        # Exact fallback for the planted synthetic manufacturer names.
        for match in KNOWN_MANUFACTURER_RE.finditer(text):
            hits.append(EvidenceHit(
                value=match.group("value").strip(" '\"."),
                file_name=file_name,
                page_number=get_page_number(rec),
                section=get_section(rec),
                chunk_id=get_chunk_id(rec),
                text_snippet=make_snippet(text, match.start(), match.end()),
            ))

        # Generic company-name extraction.
        for match in COMPANY_NAME_RE.finditer(text):
            value = match.group("value").strip(" '\".")

            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 180)
            window = text[start:end]

            has_context = bool(MANUFACTURER_CONTEXT_RE.search(window))

            if not has_context and not file_has_context:
                continue

            hits.append(EvidenceHit(
                value=value,
                file_name=file_name,
                page_number=get_page_number(rec),
                section=get_section(rec),
                chunk_id=get_chunk_id(rec),
                text_snippet=make_snippet(text, match.start(), match.end()),
            ))

    return hits


def extract_primary_endpoint_hits(records: Iterable[Dict[str, Any]]) -> List[EvidenceHit]:
    hits: List[EvidenceHit] = []

    for rec in records:
        text = get_record_text(rec)
        if not text:
            continue

        for match in PRIMARY_ENDPOINT_SENTENCE_RE.finditer(text):
            value = match.group("value").strip(" '\".")

            # Clean common trailing filler.
            value = re.sub(
                r"(?i)\s+(for|in)\s+(the\s+)?(?:study|trial|csr|clinical\s+summary).*$",
                "",
                value,
            ).strip(" '\".")

            if len(value) < 4:
                continue

            hits.append(EvidenceHit(
                value=value,
                file_name=get_file_name(rec),
                page_number=get_page_number(rec),
                section=get_section(rec),
                chunk_id=get_chunk_id(rec),
                text_snippet=make_snippet(text, match.start(), match.end()),
            ))

    return hits

def evidence_hit_to_dict(hit: EvidenceHit) -> Dict[str, Any]:
    return asdict(hit)


def hits_by_file(hits: List[EvidenceHit], canon_fn=lambda x: lower_clean(x)) -> Dict[str, List[EvidenceHit]]:
    grouped: Dict[str, List[EvidenceHit]] = defaultdict(list)
    for hit in hits:
        key = hit.file_name or "unknown_file"
        grouped[key].append(hit)
    return grouped


def unique_values(hits: List[EvidenceHit], canon_fn=lambda x: lower_clean(x)) -> Dict[str, List[EvidenceHit]]:
    values: Dict[str, List[EvidenceHit]] = defaultdict(list)
    for hit in hits:
        canon = canon_fn(hit.value)
        if canon:
            values[canon].append(hit)
    return values


def has_conflicting_values(hits: List[EvidenceHit], canon_fn=lambda x: lower_clean(x)) -> bool:
    return len(unique_values(hits, canon_fn)) >= 2


def first_evidence_values(hits: List[EvidenceHit], canon_fn=lambda x: lower_clean(x), max_values: int = 3) -> List[str]:
    values = []
    seen = set()
    for hit in hits:
        canon = canon_fn(hit.value)
        if canon and canon not in seen:
            seen.add(canon)
            values.append(hit.value)
        if len(values) >= max_values:
            break
    return values


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def load_dossier_corpus(dossier_chunks_file: Path) -> List[Dict[str, Any]]:
    chunks = read_jsonl(dossier_chunks_file)
    normalized = []
    for rec in chunks:
        text = get_record_text(rec)
        if not text:
            continue
        normalized.append(rec)
    return normalized


def package_map(packages_obj: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(packages_obj, dict):
        packages = packages_obj.get("packages", packages_obj.get("evidence_packages", packages_obj))
    else:
        packages = packages_obj

    if isinstance(packages, dict):
        iterable = packages.values()
    elif isinstance(packages, list):
        iterable = packages
    else:
        iterable = []

    out = {}
    for pkg in iterable:
        if isinstance(pkg, dict) and pkg.get("package_id"):
            out[str(pkg["package_id"])] = pkg
    return out


def collect_package_records(package: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not package:
        return []
    return list(iter_text_records(package))


# ---------------------------------------------------------------------------
# Rule checks on local package evidence
# ---------------------------------------------------------------------------

def verify_shelf_life(records: List[Dict[str, Any]]) -> Optional[RuleFinding]:
    hits = extract_hits(records, SHELF_LIFE_RE) + extract_hits(records, SHELF_LIFE_PREFIX_RE)
    if len(hits) < 2 or not has_conflicting_values(hits, lambda x: str(int(re.search(r"\d+", x).group(0))) if re.search(r"\d+", x) else ""):
        return None

    values = first_evidence_values(hits, lambda x: str(int(re.search(r"\d+", x).group(0))) if re.search(r"\d+", x) else "")
    return RuleFinding(
        rule_id="RULE-SHELF-LIFE-MISMATCH",
        rule_name="Shelf-life value mismatch",
        rule_status="confirmed",
        severity="high",
        finding_type="value_inconsistency",
        finding_summary=f"Shelf-life values conflict across retrieved dossier evidence: {', '.join(values)} months.",
        evidence_status="strong",
        evidence=[evidence_hit_to_dict(h) for h in hits[:6]],
        reviewer_action="Verify and align shelf-life information across Module 3 stability documentation and product labeling.",
    )


def verify_storage(records: List[Dict[str, Any]]) -> Optional[RuleFinding]:
    hits = extract_hits(records, STORAGE_RE)
    if len(hits) < 2 or not has_conflicting_values(hits, canonical_storage):
        return None

    values = first_evidence_values(hits, canonical_storage)
    return RuleFinding(
        rule_id="RULE-STORAGE-CONDITION-MISMATCH",
        rule_name="Storage-condition value mismatch",
        rule_status="confirmed",
        severity="high",
        finding_type="value_inconsistency",
        finding_summary=f"Storage conditions conflict across retrieved dossier evidence: {', '.join(values)}.",
        evidence_status="strong",
        evidence=[evidence_hit_to_dict(h) for h in hits[:6]],
        reviewer_action="Verify and align storage-condition information across Module 3 stability documentation and product labeling.",
    )


def verify_manufacturer(records: List[Dict[str, Any]]) -> Optional[RuleFinding]:
    hits = extract_manufacturer_hits(records)
    values = unique_values(hits, canonical_manufacturer)

    if len(values) < 2:
        return None

    raw_values = first_evidence_values(hits, canonical_manufacturer)

    return RuleFinding(
        rule_id="RULE-MANUFACTURER-MISMATCH",
        rule_name="Manufacturer/entity mismatch",
        rule_status="confirmed",
        severity="medium",
        finding_type="entity_inconsistency",
        finding_summary=f"Manufacturer names conflict across retrieved dossier evidence: {', '.join(raw_values)}.",
        evidence_status="strong",
        evidence=[evidence_hit_to_dict(h) for h in hits[:8]],
        reviewer_action="Verify and align manufacturer information across QOS and Module 3 manufacturing sections.",
    )


def verify_batch(records: List[Dict[str, Any]]) -> Optional[RuleFinding]:
    hits: List[EvidenceHit] = []
    for rec in records:
        text = get_record_text(rec)
        if not text:
            continue
        for match in BATCH_RE.finditer(text):
            hits.append(EvidenceHit(
                value=match.group(0).upper().replace(" ", "-"),
                file_name=get_file_name(rec),
                page_number=get_page_number(rec),
                section=get_section(rec),
                chunk_id=get_chunk_id(rec),
                text_snippet=make_snippet(text, match.start(), match.end()),
            ))

    values = unique_values(hits, lambda x: x.upper().replace(" ", "-"))
    if len(values) < 2:
        return None

    raw_values = first_evidence_values(hits, lambda x: x.upper().replace(" ", "-"))
    return RuleFinding(
        rule_id="RULE-BATCH-MISMATCH",
        rule_name="Batch identifier mismatch",
        rule_status="confirmed",
        severity="medium",
        finding_type="value_inconsistency",
        finding_summary=f"Batch identifiers conflict across retrieved dossier evidence: {', '.join(raw_values)}.",
        evidence_status="strong",
        evidence=[evidence_hit_to_dict(h) for h in hits[:8]],
        reviewer_action="Verify batch references across QOS, stability, manufacturing, and batch analysis sections.",
    )


def verify_endpoint(records: List[Dict[str, Any]]) -> Optional[RuleFinding]:
    hits = extract_primary_endpoint_hits(records)
    values = unique_values(hits, canonical_endpoint)

    if len(values) < 2:
        return None

    raw_values = first_evidence_values(hits, canonical_endpoint)

    return RuleFinding(
        rule_id="RULE-PRIMARY-ENDPOINT-MISMATCH",
        rule_name="Primary endpoint mismatch",
        rule_status="confirmed",
        severity="high",
        finding_type="endpoint_inconsistency",
        finding_summary=f"Primary endpoint statements conflict across retrieved dossier evidence: {', '.join(raw_values)}.",
        evidence_status="strong",
        evidence=[evidence_hit_to_dict(h) for h in hits[:8]],
        reviewer_action="Verify the primary endpoint and align the Clinical Summary with the Clinical Study Report.",
    )


def verify_sae_count(records: List[Dict[str, Any]]) -> Optional[RuleFinding]:
    hits = extract_hits(records, SAE_RE)
    values = unique_values(hits, lambda x: str(int(x)) if str(x).isdigit() else lower_clean(x))
    if len(values) < 2:
        return None

    raw_values = first_evidence_values(hits, lambda x: str(int(x)) if str(x).isdigit() else lower_clean(x))
    return RuleFinding(
        rule_id="RULE-SAE-COUNT-MISMATCH",
        rule_name="Serious adverse event count mismatch",
        rule_status="confirmed",
        severity="high",
        finding_type="safety_count_inconsistency",
        finding_summary=f"SAE counts conflict across retrieved dossier evidence: {', '.join(raw_values)}.",
        evidence_status="strong",
        evidence=[evidence_hit_to_dict(h) for h in hits[:6]],
        reviewer_action="Verify serious adverse event counts and align safety reporting across CSR and clinical summary.",
    )


def package_rule_findings(records: List[Dict[str, Any]]) -> List[RuleFinding]:
    checks = [
        verify_shelf_life,
        verify_storage,
        verify_manufacturer,
        verify_batch,
        verify_endpoint,
        verify_sae_count,
    ]
    findings = []
    for check in checks:
        finding = check(records)
        if finding:
            findings.append(finding)
    return findings


# ---------------------------------------------------------------------------
# Corpus-wide rule checks
# ---------------------------------------------------------------------------

def verify_missing_32p82(corpus_records: List[Dict[str, Any]]) -> Optional[RuleFinding]:
    all_text = "\n".join(get_record_text(r) for r in corpus_records)
    has_32p82 = bool(SECTION_32P82_RE.search(all_text))
    has_32p81 = bool(SECTION_32P81_RE.search(all_text))
    has_32p83 = bool(SECTION_32P83_RE.search(all_text))

    if has_32p82:
        return None

    if not (has_32p81 or has_32p83):
        return None

    evidence = []
    for rec in corpus_records:
        text = get_record_text(rec)
        if SECTION_32P81_RE.search(text) or SECTION_32P83_RE.search(text):
            evidence.append({
                "file_name": get_file_name(rec),
                "page_number": get_page_number(rec),
                "section": get_section(rec),
                "chunk_id": get_chunk_id(rec),
                "text_snippet": make_snippet(text, 0, min(220, len(text))),
            })
        if len(evidence) >= 4:
            break

    return RuleFinding(
        rule_id="RULE-MISSING-3.2.P.8.2",
        rule_name="Missing 3.2.P.8.2 post-approval stability protocol",
        rule_status="rule_detected_gap",
        severity="high",
        finding_type="missing_section",
        finding_summary="Dossier evidence includes nearby stability sections but no detectable 3.2.P.8.2 post-approval stability protocol or stability commitment.",
        evidence_status="strong" if evidence else "medium",
        evidence=evidence,
        reviewer_action="Verify whether 3.2.P.8.2 exists elsewhere or add the post-approval stability protocol and stability commitment.",
    )   


def verify_csr_synopsis_missing(corpus_records: List[Dict[str, Any]]) -> Optional[RuleFinding]:
    csr_records = []
    for rec in corpus_records:
        file_name = lower_clean(get_file_name(rec) or "")
        section = lower_clean(get_section(rec) or "")
        text = lower_clean(get_record_text(rec))
        if "clinical_study_report" in file_name or "clinical study report" in section or "module 5" in section:
            csr_records.append(rec)

    if not csr_records:
        return None

    csr_text = "\n".join(get_record_text(r) for r in csr_records)
    if SYNOPSIS_RE.search(csr_text):
        return None

    evidence = []
    for rec in csr_records[:5]:
        text = get_record_text(rec)
        evidence.append({
            "file_name": get_file_name(rec),
            "page_number": get_page_number(rec),
            "section": get_section(rec),
            "chunk_id": get_chunk_id(rec),
            "text_snippet": make_snippet(text, 0, min(220, len(text))),
        })

    return RuleFinding(
        rule_id="RULE-MISSING-CSR-SYNOPSIS",
        rule_name="Missing CSR synopsis",
        rule_status="rule_detected_gap",
        severity="high",
        finding_type="missing_section",
        finding_summary="Clinical Study Report records were found, but no separate Synopsis section was detected.",
        evidence_status="medium",
        evidence=evidence,
        reviewer_action="Verify whether a CSR synopsis exists or add the missing synopsis section.",
    )


def corpus_rule_findings(corpus_records: List[Dict[str, Any]]) -> List[RuleFinding]:
    findings = []

    # Global deterministic mismatches can be found by applying package-level
    # checks over the full dossier corpus.
    for check in [
        verify_shelf_life,
        verify_storage,
        verify_manufacturer,
        verify_batch,
        verify_endpoint,
        verify_sae_count,
    ]:
        finding = check(corpus_records)
        if finding:
            finding.rule_status = "rule_detected_gap"
            findings.append(finding)

    for check in [verify_missing_32p82, verify_csr_synopsis_missing]:
        finding = check(corpus_records)
        if finding:
            findings.append(finding)

    return findings


# ---------------------------------------------------------------------------
# LLM decision reconciliation prep
# ---------------------------------------------------------------------------

def infer_expected_rule_type(decision: Dict[str, Any]) -> Optional[str]:
    finding_type = lower_clean(decision.get("finding_type"))
    trace_type = lower_clean(decision.get("trace", {}).get("fact_type"))
    summary = lower_clean(decision.get("finding_summary", "") + " " + decision.get("reasoning_summary", ""))

    if "shelf" in trace_type or "shelf" in summary:
        return "RULE-SHELF-LIFE-MISMATCH"
    if "storage" in trace_type or "storage" in summary or "store below" in summary:
        return "RULE-STORAGE-CONDITION-MISMATCH"
    if "manufacturer" in trace_type or "manufacturer" in summary:
        return "RULE-MANUFACTURER-MISMATCH"
    if "batch" in trace_type or "batch" in summary:
        return "RULE-BATCH-MISMATCH"
    if "endpoint" in trace_type or "endpoint" in summary or finding_type == "endpoint_inconsistency":
        return "RULE-PRIMARY-ENDPOINT-MISMATCH"
    if "sae" in trace_type or "serious adverse" in summary or finding_type == "safety_count_inconsistency":
        return "RULE-SAE-COUNT-MISMATCH"
    if "3.2.p.8.2" in summary or finding_type == "missing_section":
        return "RULE-MISSING-3.2.P.8.2"
    return None


def summarize_llm_rule_result(decision: Dict[str, Any], findings: List[RuleFinding]) -> Dict[str, Any]:
    expected_rule = infer_expected_rule_type(decision)
    llm_decision = lower_clean(decision.get("llm_decision"))
    llm_finding_type = lower_clean(decision.get("finding_type"))

    matched = None
    if expected_rule:
        matched = next((f for f in findings if f.rule_id == expected_rule), None)

    if matched:
        verifier_status = "confirmed" if llm_decision in {"gap", "uncertain", "needs_human_review"} else "rule_detected_gap"
        return {
            "package_id": decision.get("package_id"),
            "llm_decision": decision.get("llm_decision"),
            "llm_finding_type": decision.get("finding_type"),
            "expected_rule_id": expected_rule,
            "verifier_status": verifier_status,
            "matched_rule_finding": asdict(matched),
        }

    if expected_rule:
        # The LLM made a deterministic claim, but rule evidence in this package did not confirm it.
        return {
            "package_id": decision.get("package_id"),
            "llm_decision": decision.get("llm_decision"),
            "llm_finding_type": decision.get("finding_type"),
            "expected_rule_id": expected_rule,
            "verifier_status": "not_applicable" if llm_finding_type not in DETERMINISTIC_FINDING_TYPES else "contradicted",
            "matched_rule_finding": None,
        }

    if findings and llm_decision == "no_gap":
        return {
            "package_id": decision.get("package_id"),
            "llm_decision": decision.get("llm_decision"),
            "llm_finding_type": decision.get("finding_type"),
            "expected_rule_id": None,
            "verifier_status": "rule_detected_gap",
            "matched_rule_finding": asdict(findings[0]),
        }

    if findings:
        return {
            "package_id": decision.get("package_id"),
            "llm_decision": decision.get("llm_decision"),
            "llm_finding_type": decision.get("finding_type"),
            "expected_rule_id": None,
            "verifier_status": "confirmed" if llm_decision in {"gap", "uncertain", "needs_human_review"} else "rule_detected_gap",
            "matched_rule_finding": asdict(findings[0]),
        }

    return {
        "package_id": decision.get("package_id"),
        "llm_decision": decision.get("llm_decision"),
        "llm_finding_type": decision.get("finding_type"),
        "expected_rule_id": expected_rule,
        "verifier_status": "passed" if llm_decision == "no_gap" else "not_applicable",
        "matched_rule_finding": None,
    }


# ---------------------------------------------------------------------------
# Main verification runner
# ---------------------------------------------------------------------------

def run_rule_verification(
    llm_decisions_file: Path = DEFAULT_LLM_DECISIONS,
    evidence_packages_file: Path = DEFAULT_EVIDENCE_PACKAGES,
    dossier_chunks_file: Path = DEFAULT_DOSSIER_CHUNKS,
    output_file: Path = DEFAULT_OUTPUT,
) -> Dict[str, Any]:
    llm_obj = read_json(llm_decisions_file)
    decisions = llm_obj.get("decisions", [])

    packages_obj = read_json(evidence_packages_file) if evidence_packages_file.exists() else {}
    packages = package_map(packages_obj)
    dossier_corpus = load_dossier_corpus(dossier_chunks_file)

    per_decision_results = []
    all_package_findings: List[RuleFinding] = []

    for decision in decisions:
        pkg_id = str(decision.get("package_id"))
        records = collect_package_records(packages.get(pkg_id))

        # Fallback: if the package cannot be found, use citation metadata only.
        if not records:
            records = []
            for cite in decision.get("dossier_citations", []):
                records.append({
                    "file_name": cite.get("file_name"),
                    "page_number": cite.get("page_number"),
                    "section": cite.get("section"),
                    "chunk_id": cite.get("chunk_id"),
                    "text": "",
                })

        findings = package_rule_findings(records)
        all_package_findings.extend(findings)
        result = summarize_llm_rule_result(decision, findings)
        result["package_rule_findings"] = [asdict(f) for f in findings]
        per_decision_results.append(result)

    corpus_findings = corpus_rule_findings(dossier_corpus)

    status_counts = Counter(r["verifier_status"] for r in per_decision_results)
    rule_counts = Counter(f.rule_id for f in all_package_findings + corpus_findings)

    output = {
        "run_type": "rule_verification_layer",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_decisions_file": str(llm_decisions_file),
        "input_evidence_packages_file": str(evidence_packages_file),
        "input_dossier_chunks_file": str(dossier_chunks_file),
        "decision_count": len(decisions),
        "per_decision_results": per_decision_results,
        "corpus_rule_findings": [asdict(f) for f in corpus_findings],
        "counts": {
            "verifier_status": dict(status_counts),
            "rule_id": dict(rule_counts),
            "corpus_rule_findings": len(corpus_findings),
            "package_rule_findings": len(all_package_findings),
        },
        "note": (
            "Rule verification checks deterministic evidence only. Unsupported claims and weak justifications "
            "may require LLM/human review and are often marked not_applicable by the rule layer."
        ),
    }

    ensure_parent(output_file)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic rule verification over LLM decisions.")
    parser.add_argument("--llm-decisions", type=Path, default=DEFAULT_LLM_DECISIONS)
    parser.add_argument("--evidence-packages", type=Path, default=DEFAULT_EVIDENCE_PACKAGES)
    parser.add_argument("--dossier-chunks", type=Path, default=DEFAULT_DOSSIER_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output = run_rule_verification(
        llm_decisions_file=args.llm_decisions,
        evidence_packages_file=args.evidence_packages,
        dossier_chunks_file=args.dossier_chunks,
        output_file=args.output,
    )

    print("Rule verification complete")
    print(f"Decisions checked: {output['decision_count']}")
    print(f"Output written to: {args.output}")
    print("Verifier status counts:")
    for key, value in output["counts"]["verifier_status"].items():
        print(f"  {key}: {value}")
    print("Corpus rule findings:")
    for finding in output["corpus_rule_findings"]:
        print(f"  {finding['rule_id']} | {finding['severity']} | {finding['finding_summary']}")


if __name__ == "__main__":
    main()
