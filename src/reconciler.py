"""
src/reconciler.py

Combines LLM decisions with deterministic rule-verification findings.

Purpose:
    LLM decision layer = primary judgment
    Rule verifier      = deterministic double-check
    Reconciler         = final status assignment

Default inputs:
    outputs/llm_decisions.json
    outputs/rule_verification_results.json

Default output:
    outputs/reconciled_gap_report.json

Run:
    python -m src.reconciler

Recommended with saved v2 decisions:
    python -m src.reconciler --llm-decisions outputs/llm_decisions_v2_d2g.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_LLM_DECISIONS = Path("outputs/llm_decisions.json")
DEFAULT_RULE_RESULTS = Path("outputs/rule_verification_results.json")
DEFAULT_OUTPUT = Path("outputs/reconciled_gap_report.json")

RECONCILIATION_POLICY_VERSION = "v1.0"

FINAL_STATUS_ORDER = [
    "confirmed_gap",
    "rule_flagged_gap",
    "potential_gap",
    "needs_human_review",
    "unconfirmed_llm_flag",
    "no_gap_detected",
]

SEVERITY_ORDER = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "none": 0,
    "": 0,
    None: 0,
}

RULE_ID_TO_EXPECTED_FINDING_TYPE = {
    "RULE-SHELF-LIFE-MISMATCH": "value_inconsistency",
    "RULE-STORAGE-CONDITION-MISMATCH": "value_inconsistency",
    "RULE-MANUFACTURER-MISMATCH": "entity_inconsistency",
    "RULE-BATCH-MISMATCH": "value_inconsistency",
    "RULE-PRIMARY-ENDPOINT-MISMATCH": "endpoint_inconsistency",
    "RULE-SAE-COUNT-MISMATCH": "safety_count_inconsistency",
    "RULE-MISSING-3.2.P.8.2": "missing_section",
    "RULE-MISSING-CSR-SYNOPSIS": "missing_section",
}

FACT_TYPE_TO_RULE_HINT = {
    "shelf_life_statement": "RULE-SHELF-LIFE-MISMATCH",
    "storage_condition_statement": "RULE-STORAGE-CONDITION-MISMATCH",
    "manufacturer_statement": "RULE-MANUFACTURER-MISMATCH",
    "batch_statement": "RULE-BATCH-MISMATCH",
    "primary_endpoint_statement": "RULE-PRIMARY-ENDPOINT-MISMATCH",
    "endpoint_statement": "RULE-PRIMARY-ENDPOINT-MISMATCH",
    "sae_count_statement": "RULE-SAE-COUNT-MISMATCH",
    "safety_count_statement": "RULE-SAE-COUNT-MISMATCH",
}

KEYWORD_TO_RULE_HINT = [
    ("shelf", "RULE-SHELF-LIFE-MISMATCH"),
    ("storage", "RULE-STORAGE-CONDITION-MISMATCH"),
    ("store below", "RULE-STORAGE-CONDITION-MISMATCH"),
    ("manufacturer", "RULE-MANUFACTURER-MISMATCH"),
    ("manufacturing", "RULE-MANUFACTURER-MISMATCH"),
    ("batch", "RULE-BATCH-MISMATCH"),
    ("endpoint", "RULE-PRIMARY-ENDPOINT-MISMATCH"),
    ("serious adverse", "RULE-SAE-COUNT-MISMATCH"),
    ("sae", "RULE-SAE-COUNT-MISMATCH"),
    ("3.2.p.8.2", "RULE-MISSING-3.2.P.8.2"),
    ("synopsis", "RULE-MISSING-CSR-SYNOPSIS"),
]


@dataclass
class ReconciledFinding:
    finding_id: str
    source: str
    package_id: Optional[str]
    rule_id: Optional[str]
    llm_decision: str
    llm_finding_type: str
    rule_status: str
    final_status: str
    severity: str
    evidence_status: str
    finding_type: str
    finding_summary: str
    reasoning_summary: str
    reviewer_action: str
    guideline_citations: List[Dict[str, Any]]
    dossier_citations: List[Dict[str, Any]]
    rule_evidence: List[Dict[str, Any]]
    reconciliation_reason: str


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_llm_decision(value: Any) -> str:
    text = normalize_text(value).lower()
    if text in {"gap", "inconsistency"}:
        return "gap"
    if text in {"no_gap", "no gap", "none"}:
        return "no_gap"
    if text in {"uncertain", "unknown"}:
        return "uncertain"
    if text in {"needs_human_review", "human_review", "insufficient_evidence"}:
        return "needs_human_review"
    return text or "uncertain"


def normalize_finding_type(value: Any) -> str:
    text = normalize_text(value).lower()
    mapping = {
        "name_mismatch": "entity_inconsistency",
        "count_mismatch": "safety_count_inconsistency",
        "endpoint_name_mismatch": "endpoint_inconsistency",
        "truncated_evidence": "insufficient_evidence",
        "none": "no_gap",
    }
    return mapping.get(text, text or "other")


def normalize_severity(value: Any) -> str:
    text = normalize_text(value).lower()
    mapping = {
        "critical": "high",
        "major": "medium",
        "minor": "low",
        "informational": "none",
        "": "none",
    }
    return mapping.get(text, text if text in {"high", "medium", "low", "none"} else "low")


def normalize_evidence_status(value: Any) -> str:
    text = normalize_text(value).lower()
    if text in {"strong", "medium", "weak", "missing"}:
        return text
    if text in {"none", "absent", "not_found"}:
        return "missing"
    return "weak"


def evidence_bucket(value: str) -> str:
    status = normalize_evidence_status(value)
    if status == "strong":
        return "strong"
    if status == "missing":
        return "missing"
    return "weak"


def normalize_rule_status(value: Any) -> str:
    text = normalize_text(value).lower()
    if text in {"confirmed", "rule_detected_gap", "detected", "rule_detected"}:
        return "confirmed"
    if text in {"contradicted", "failed"}:
        return "contradicted"
    if text in {"passed", "pass"}:
        return "passed"
    if text in {"not_applicable", "not applicable", "na", "n/a", ""}:
        return "not_applicable"
    return text


def max_severity(*values: Any) -> str:
    normalized = [normalize_severity(v) for v in values]
    return max(normalized, key=lambda s: SEVERITY_ORDER.get(s, 0)) if normalized else "none"


def status_counts(findings: Iterable[ReconciledFinding]) -> Dict[str, int]:
    return dict(Counter(f.final_status for f in findings))


def severity_counts(findings: Iterable[ReconciledFinding]) -> Dict[str, int]:
    return dict(Counter(f.severity for f in findings))


def finding_type_counts(findings: Iterable[ReconciledFinding]) -> Dict[str, int]:
    return dict(Counter(f.finding_type for f in findings))


def extract_decisions(llm_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    decisions = llm_obj.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("LLM decisions file must contain a list at key 'decisions'.")
    return decisions


def extract_corpus_rule_findings(rule_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["corpus_rule_findings", "rule_findings", "findings"]:
        value = rule_obj.get(key)
        if isinstance(value, list):
            return value
    return []


def extract_per_decision_results(rule_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in [
        "per_decision_results",
        "decision_rule_results",
        "per_package_results",
        "package_rule_results",
        "decision_results",
    ]:
        value = rule_obj.get(key)
        if isinstance(value, list):
            return value
    return []


def build_per_decision_map(rule_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in extract_per_decision_results(rule_obj):
        package_id = normalize_text(item.get("package_id"))
        if package_id:
            out[package_id] = item
    return out


def infer_rule_hint_from_decision(decision: Dict[str, Any]) -> Optional[str]:
    trace = decision.get("trace", {}) if isinstance(decision.get("trace"), dict) else {}
    fact_type = normalize_text(trace.get("fact_type") or decision.get("fact_type")).lower()
    if fact_type in FACT_TYPE_TO_RULE_HINT:
        return FACT_TYPE_TO_RULE_HINT[fact_type]

    combined = " ".join(
        normalize_text(decision.get(k)).lower()
        for k in ["finding_type", "finding_summary", "reasoning_summary", "ctd_section"]
    )
    combined += " " + normalize_text(trace.get("source_file")).lower()

    for keyword, rule_id in KEYWORD_TO_RULE_HINT:
        if keyword in combined:
            return rule_id
    return None


def find_matching_rule_finding(
    decision: Dict[str, Any],
    corpus_findings: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    finding_type = normalize_finding_type(decision.get("finding_type"))

    decision_text = normalize_text(
        " ".join([
            str(decision.get("finding_summary") or ""),
            str(decision.get("reasoning_summary") or ""),
            str(decision.get("reviewer_action") or ""),
            str(decision.get("ctd_section") or ""),
        ])
    ) or ""

    rule_id_hint = None

    if "shelf" in decision_text or "18 months" in decision_text or "24 months" in decision_text:
        rule_id_hint = "RULE-SHELF-LIFE-MISMATCH"

    elif "storage" in decision_text or "store below" in decision_text or "30" in decision_text and "25" in decision_text:
        rule_id_hint = "RULE-STORAGE-CONDITION-MISMATCH"

    elif "manufacturer" in decision_text or "apex" in decision_text or "nova" in decision_text:
        rule_id_hint = "RULE-MANUFACTURER-MISMATCH"

    elif "batch" in decision_text or "b003" in decision_text or "b001" in decision_text or "b002" in decision_text:
        rule_id_hint = "RULE-BATCH-MISMATCH"

    elif "sae" in decision_text or "serious adverse" in decision_text:
        rule_id_hint = "RULE-SAE-COUNT-MISMATCH"

    elif "3.2.p.8.2" in decision_text or "post-approval stability" in decision_text:
        rule_id_hint = "RULE-MISSING-3.2.P.8.2"

    elif "csr synopsis" in decision_text or "synopsis" in decision_text:
        rule_id_hint = "RULE-MISSING-CSR-SYNOPSIS"

    elif "primary endpoint" in decision_text:
        rule_id_hint = "RULE-PRIMARY-ENDPOINT-MISMATCH"

    # Prevent label claims from matching the primary-endpoint rule.
    label_claim_markers = [
        "label claim",
        "claim differs",
        "rapid control",
        "within 24 hours",
        "unsupported claim",
        "label claims",
        "claim lacks",
    ]

    is_label_claim_case = any(marker in decision_text for marker in label_claim_markers)

    if is_label_claim_case:
        if rule_id_hint == "RULE-PRIMARY-ENDPOINT-MISMATCH":
            rule_id_hint = None

    if rule_id_hint:
        for finding in corpus_findings:
            if normalize_text(finding.get("rule_id")) == rule_id_hint:
                return finding

    # Fallback: match by finding_type, but never match unsupported claims to endpoint rule.
    for finding in corpus_findings:
        rule_id = normalize_text(finding.get("rule_id"))
        rule_type = normalize_finding_type(finding.get("finding_type"))

        if is_label_claim_case and rule_id == "RULE-PRIMARY-ENDPOINT-MISMATCH":
            continue

        if finding_type == rule_type:
            return finding

    return None

def reconcile_status(llm_decision: str, evidence_status: str, rule_status: str):
    llm_decision = normalize_llm_decision(llm_decision)
    evidence_status = normalize_evidence_status(evidence_status)
    rule_status = normalize_rule_status(rule_status)

    # If deterministic rule confirms the same issue, upgrade to confirmed gap.
    if rule_status in {"confirmed", "rule_detected_gap"}:
        if llm_decision == "no_gap":
            return (
                "rule_flagged_gap",
                "Rule verifier detected a deterministic gap even though the LLM did not flag it.",
            )

        return (
            "confirmed_gap",
            "LLM finding is supported by deterministic rule verification.",
        )

    if rule_status == "contradicted":
        return (
            "unconfirmed_llm_flag",
            "LLM finding was contradicted by deterministic rule verification.",
        )

    if llm_decision == "gap":
        if evidence_status in {"strong", "medium", "weak"}:
            return (
                "potential_gap",
                "LLM identified a potential gap, but no matching deterministic rule confirmed it.",
            )

        return (
            "unconfirmed_llm_flag",
            "LLM identified a gap but evidence support was missing or insufficient.",
        )

    if llm_decision == "uncertain":
        return (
            "needs_human_review",
            "LLM was uncertain and no deterministic rule confirmed the issue.",
        )

    if llm_decision == "needs_human_review":
        return (
            "needs_human_review",
            "LLM requested human review.",
        )

    if llm_decision == "no_gap":
        return (
            "no_gap_detected",
            "LLM did not identify a gap and no deterministic rule contradicted it.",
        )

    return (
        "needs_human_review",
        "Decision could not be reconciled automatically.",
    )


def rule_finding_to_reconciled(
    rule_finding: Dict[str, Any],
    index: int,
    source: str = "rule_only",
) -> ReconciledFinding:
    rule_id = normalize_text(rule_finding.get("rule_id"))
    severity = normalize_severity(rule_finding.get("severity"))
    finding_type = normalize_finding_type(
        rule_finding.get("finding_type") or RULE_ID_TO_EXPECTED_FINDING_TYPE.get(rule_id, "other")
    )
    summary = normalize_text(rule_finding.get("finding_summary"))
    reviewer_action = normalize_text(rule_finding.get("reviewer_action")) or "Review and resolve the rule-flagged dossier inconsistency."

    return ReconciledFinding(
        finding_id=f"REC-RULE-{index:04d}",
        source=source,
        package_id=None,
        rule_id=rule_id or None,
        llm_decision="not_applicable",
        llm_finding_type="not_applicable",
        rule_status="confirmed",
        final_status="rule_flagged_gap",
        severity=severity,
        evidence_status=normalize_evidence_status(rule_finding.get("evidence_status") or "strong"),
        finding_type=finding_type,
        finding_summary=summary,
        reasoning_summary="Detected by deterministic rule verification across the dossier corpus.",
        reviewer_action=reviewer_action,
        guideline_citations=[],
        dossier_citations=[],
        rule_evidence=rule_finding.get("evidence", []) if isinstance(rule_finding.get("evidence"), list) else [],
        reconciliation_reason="Independent rule finding added as a rule-flagged gap.",
    )


def decision_to_reconciled(
    decision: Dict[str, Any],
    index: int,
    per_decision_rule: Optional[Dict[str, Any]],
    matching_rule: Optional[Dict[str, Any]],
) -> ReconciledFinding:
    package_id = normalize_text(decision.get("package_id")) or None

    llm_decision = normalize_llm_decision(decision.get("llm_decision"))
    llm_finding_type = normalize_finding_type(decision.get("finding_type"))
    evidence_status = normalize_evidence_status(decision.get("evidence_status"))

    decision_text = normalize_text(
        " ".join([
            str(decision.get("finding_summary") or ""),
            str(decision.get("reasoning_summary") or ""),
            str(decision.get("reviewer_action") or ""),
            str(decision.get("ctd_section") or ""),
        ])
    ) or ""

    finding_type = llm_finding_type

    label_claim_markers = [
        "label claim",
        "claim differs",
        "rapid control",
        "within 24 hours",
        "unsupported claim",
        "label claims",
        "claim lacks",
    ]

    is_label_claim_case = any(marker in decision_text for marker in label_claim_markers)

    # Correct common LLM mistake:
    # Label claim support issues are unsupported_claim, not endpoint_inconsistency.
    if finding_type == "endpoint_inconsistency" and is_label_claim_case:
        if "primary endpoint" not in decision_text:
            finding_type = "unsupported_claim"

            # Important:
            # Do not let a label-claim case get confirmed by the primary-endpoint rule.
            if matching_rule:
                matched_rule_id = normalize_text(matching_rule.get("rule_id"))
                if matched_rule_id == "RULE-PRIMARY-ENDPOINT-MISMATCH":
                    matching_rule = None

    rule_id = None
    rule_status = "not_applicable"
    rule_severity = "none"
    rule_evidence: List[Dict[str, Any]] = []

    if per_decision_rule:
        rule_status = normalize_rule_status(
            per_decision_rule.get("verifier_status") or per_decision_rule.get("rule_status")
        )
        rule_id = normalize_text(
            per_decision_rule.get("expected_rule_id") or per_decision_rule.get("rule_id")
        ) or None

        matched = per_decision_rule.get("matched_rule_finding") or per_decision_rule.get("rule_finding")
        if isinstance(matched, dict):
            matching_rule = matched

    # Apply the same endpoint guard again after per_decision_rule handling.
    if finding_type == "unsupported_claim" and matching_rule:
        matched_rule_id = normalize_text(matching_rule.get("rule_id"))
        if matched_rule_id == "RULE-PRIMARY-ENDPOINT-MISMATCH":
            matching_rule = None
            rule_id = None
            rule_status = "not_applicable"

    if matching_rule:
        rule_id = normalize_text(matching_rule.get("rule_id")) or rule_id
        rule_status = "confirmed"
        rule_severity = normalize_severity(matching_rule.get("severity"))
        rule_evidence = matching_rule.get("evidence", []) if isinstance(matching_rule.get("evidence"), list) else []

    final_status, reason = reconcile_status(llm_decision, evidence_status, rule_status)

        # If rule confirms the finding, use the rule's finding_type.
    if matching_rule and final_status in {"confirmed_gap", "rule_flagged_gap"}:
        rule_type = normalize_finding_type(matching_rule.get("finding_type"))
        if rule_type:
            finding_type = rule_type

    # Unsupported label claims without deterministic rule confirmation
    # should remain potential gaps, not unconfirmed flags.
    if finding_type == "unsupported_claim" and not matching_rule:
        if llm_decision == "gap":
            final_status = "potential_gap"
            reason = "LLM identified an unsupported claim requiring reviewer confirmation."

    final_severity = max_severity(decision.get("severity"), rule_severity)

    # If no rule matched and the corrected type is unsupported_claim,
    # keep the LLM severity/action rather than endpoint-rule severity/action.
    llm_summary = normalize_text(decision.get("finding_summary")) or ""
    rule_summary = normalize_text(matching_rule.get("finding_summary")) if matching_rule else ""
    rule_summary = rule_summary or ""

    if final_status in {"confirmed_gap", "rule_flagged_gap"} and rule_summary:
        finding_summary = rule_summary
    else:
        finding_summary = llm_summary

    if not finding_summary:
        finding_summary = "Reconciled finding generated from LLM and rule-verifier outputs."

    llm_action = normalize_text(decision.get("reviewer_action")) or ""
    rule_action = normalize_text(matching_rule.get("reviewer_action")) if matching_rule else ""
    rule_action = rule_action or ""

    reviewer_action = rule_action or llm_action or "Review the finding and resolve the dossier inconsistency."

    return ReconciledFinding(
        finding_id=f"REC-LLM-{index:04d}",
        source="llm_plus_rule" if matching_rule or per_decision_rule else "llm_only",
        package_id=package_id,
        rule_id=rule_id,
        llm_decision=llm_decision,
        llm_finding_type=llm_finding_type,
        rule_status=rule_status,
        final_status=final_status,
        severity=final_severity,
        evidence_status=evidence_status,
        finding_type=finding_type,
        finding_summary=finding_summary,
        reasoning_summary=normalize_text(decision.get("reasoning_summary")) or "",
        reviewer_action=reviewer_action,
        guideline_citations=decision.get("guideline_citations", []) if isinstance(decision.get("guideline_citations"), list) else [],
        dossier_citations=decision.get("dossier_citations", []) if isinstance(decision.get("dossier_citations"), list) else [],
        rule_evidence=rule_evidence,
        reconciliation_reason=reason,
    )

def rule_signature(rule_finding: Dict[str, Any]) -> str:
    rule_id = normalize_text(rule_finding.get("rule_id"))
    summary = normalize_text(rule_finding.get("finding_summary")).lower()
    summary = re.sub(r"\s+", " ", summary)
    return f"{rule_id}|{summary[:160]}"


def run_reconciliation(
    llm_decisions_file: Path = DEFAULT_LLM_DECISIONS,
    rule_results_file: Path = DEFAULT_RULE_RESULTS,
    output_file: Path = DEFAULT_OUTPUT,
) -> Dict[str, Any]:
    llm_obj = read_json(llm_decisions_file)
    rule_obj = read_json(rule_results_file)

    decisions = extract_decisions(llm_obj)
    corpus_rules = extract_corpus_rule_findings(rule_obj)
    per_decision_map = build_per_decision_map(rule_obj)

    reconciled: List[ReconciledFinding] = []
    used_rule_signatures = set()

    for idx, decision in enumerate(decisions, start=1):
        package_id = normalize_text(decision.get("package_id"))
        per_decision_rule = per_decision_map.get(package_id)
        matching_rule = find_matching_rule_finding(decision, corpus_rules)
        if matching_rule:
            used_rule_signatures.add(rule_signature(matching_rule))
        reconciled.append(decision_to_reconciled(decision, idx, per_decision_rule, matching_rule))

    rule_only_count = 0
    for rule_finding in corpus_rules:
        sig = rule_signature(rule_finding)
        if sig in used_rule_signatures:
            continue
        rule_only_count += 1
        reconciled.append(rule_finding_to_reconciled(rule_finding, rule_only_count))

    final_findings = [asdict(f) for f in reconciled]

    report = {
        "report_type": "reconciled_gap_report",
        "reconciliation_policy_version": RECONCILIATION_POLICY_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "llm_decisions_file": str(llm_decisions_file),
            "rule_results_file": str(rule_results_file),
        },
        "summary": {
            "llm_decision_count": len(decisions),
            "corpus_rule_finding_count": len(corpus_rules),
            "final_finding_count": len(final_findings),
            "final_status_counts": status_counts(reconciled),
            "severity_counts": severity_counts(reconciled),
            "finding_type_counts": finding_type_counts(reconciled),
        },
        "policy_notes": [
            "LLM decisions are treated as primary judgments.",
            "Deterministic rule findings can confirm, contradict, or add rule-flagged gaps.",
            "Final status is reviewer-assistance output, not regulatory approval or rejection.",
        ],
        "final_findings": final_findings,
    }

    write_json(report, output_file)
    return report


def print_report_summary(report: Dict[str, Any], output_file: Path) -> None:
    summary = report.get("summary", {})
    print("Reconciliation complete")
    print(f"Output written to: {output_file}")
    print(f"LLM decisions: {summary.get('llm_decision_count', 0)}")
    print(f"Corpus rule findings: {summary.get('corpus_rule_finding_count', 0)}")
    print(f"Final findings: {summary.get('final_finding_count', 0)}")

    print("Final status counts:")
    for key, value in summary.get("final_status_counts", {}).items():
        print(f"  {key}: {value}")

    print("\nFinal findings:")
    for finding in report.get("final_findings", []):
        print(
            f"{finding['finding_id']} | {finding['final_status']} | "
            f"{finding['severity']} | {finding['finding_type']} | "
            f"{finding['finding_summary']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile LLM decisions with rule verifier results.")
    parser.add_argument("--llm-decisions", type=Path, default=DEFAULT_LLM_DECISIONS)
    parser.add_argument("--rule-results", type=Path, default=DEFAULT_RULE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_reconciliation(
        llm_decisions_file=args.llm_decisions,
        rule_results_file=args.rule_results,
        output_file=args.output,
    )
    print_report_summary(report, args.output)


if __name__ == "__main__":
    main()
