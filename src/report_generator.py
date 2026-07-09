"""
src/report_generator.py

Generate reviewer-friendly gap reports from reconciled LLM + rule-verifier output.

Default input:
    outputs/reconciled_gap_report.json

Default outputs:
    outputs/gap_report.md
    outputs/gap_report.json
    outputs/gap_report.csv

Run:
    python -m src.report_generator

Run with explicit files:
    python -m src.report_generator --input outputs/reconciled_gap_report.json --markdown outputs/gap_report.md --json outputs/gap_report.json --csv outputs/gap_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT = Path("outputs/reconciled_gap_report.json")
DEFAULT_MARKDOWN_OUTPUT = Path("outputs/gap_report.md")
DEFAULT_JSON_OUTPUT = Path("outputs/gap_report.json")
DEFAULT_CSV_OUTPUT = Path("outputs/gap_report.csv")

PROJECT_TITLE = "LLM-First Dual-RAG Regulatory Dossier Gap Assistant"
DISCLAIMER = (
    "Reviewer-assistance prototype only. This report does not certify regulatory "
    "compliance and does not approve or reject a pharmaceutical product. Final "
    "regulatory judgment remains with qualified human reviewers."
)

STATUS_ORDER = {
    "confirmed_gap": 0,
    "rule_flagged_gap": 1,
    "potential_gap": 2,
    "needs_human_review": 3,
    "unconfirmed_llm_flag": 4,
    "no_gap_detected": 5,
}

SEVERITY_ORDER = {
    "high": 0,
    "medium": 1,
    "low": 2,
    "none": 3,
}

REPORTABLE_STATUSES = {
    "confirmed_gap",
    "rule_flagged_gap",
    "potential_gap",
    "needs_human_review",
    "unconfirmed_llm_flag",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_status(value: Any) -> str:
    return as_text(value, "needs_human_review").lower().strip()


def normalize_severity(value: Any) -> str:
    value = as_text(value, "none").lower().strip()
    if value in {"high", "medium", "low", "none"}:
        return value
    if value in {"critical", "major"}:
        return "high" if value == "critical" else "medium"
    if value in {"minor", "informational"}:
        return "low" if value == "minor" else "none"
    return "none"


def normalize_finding_type(value: Any) -> str:
    return as_text(value, "other").lower().strip()


def status_label(status: str) -> str:
    labels = {
        "confirmed_gap": "Confirmed Gap",
        "rule_flagged_gap": "Rule-Flagged Gap",
        "potential_gap": "Potential Gap",
        "needs_human_review": "Needs Human Review",
        "unconfirmed_llm_flag": "Unconfirmed LLM Flag",
        "no_gap_detected": "No Gap Detected",
    }
    return labels.get(status, status.replace("_", " ").title())


def severity_label(severity: str) -> str:
    return severity.upper() if severity != "none" else "None"


def md_escape(text: Any) -> str:
    value = as_text(text, "")
    value = value.replace("|", "\\|")
    value = value.replace("\n", " ")
    return value


def citation_key(citation: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        as_text(citation.get("guideline_id") or citation.get("file_name") or citation.get("source_file")),
        as_text(citation.get("source_file") or citation.get("file_name")),
        as_text(citation.get("page_number")),
        as_text(citation.get("section")),
        as_text(citation.get("chunk_id")),
    )


def dedupe_dicts(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = citation_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def merge_severity(values: Iterable[Any]) -> str:
    severities = [normalize_severity(v) for v in values]
    if not severities:
        return "none"
    return min(severities, key=lambda s: SEVERITY_ORDER.get(s, 99))


def make_unsupported_claim_group(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = deepcopy(findings[0])
    first["finding_id"] = "REC-GROUP-UNSUPPORTED-CLAIM"
    first["source"] = "grouped_llm_findings"
    first["final_status"] = "potential_gap"
    first["finding_type"] = "unsupported_claim"
    first["severity"] = merge_severity(f.get("severity") for f in findings)
    first["finding_summary"] = (
        f"Unsupported label claim requires reviewer confirmation. "
        f"Consolidated from {len(findings)} related LLM finding(s)."
    )
    first["reasoning_summary"] = (
        "Multiple retrieved label/clinical evidence packages indicate that a label claim may lack "
        "direct supporting clinical evidence. The finding is grouped to avoid duplicate report rows."
    )
    first["reviewer_action"] = (
        "Verify whether the label claim is supported by Module 5 clinical evidence. "
        "If not supported, revise or remove the claim."
    )
    first["grouped_finding_ids"] = [as_text(f.get("finding_id")) for f in findings]
    first["guideline_citations"] = dedupe_dicts(
        citation for f in findings for citation in f.get("guideline_citations", [])
    )
    first["dossier_citations"] = dedupe_dicts(
        citation for f in findings for citation in f.get("dossier_citations", [])
    )
    first["rule_evidence"] = dedupe_dicts(
        evidence for f in findings for evidence in f.get("rule_evidence", [])
    )
    return first


def deduplicate_findings(findings: List[Dict[str, Any]], dedupe_unsupported_claims: bool = True) -> List[Dict[str, Any]]:
    if not dedupe_unsupported_claims:
        return findings

    unsupported: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []

    for finding in findings:
        status = normalize_status(finding.get("final_status"))
        ftype = normalize_finding_type(finding.get("finding_type"))
        if ftype == "unsupported_claim" and status in {"potential_gap", "unconfirmed_llm_flag", "needs_human_review"}:
            unsupported.append(finding)
        else:
            others.append(finding)

    if len(unsupported) <= 1:
        return findings

    grouped = make_unsupported_claim_group(unsupported)
    return others + [grouped]


def sort_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(finding: Dict[str, Any]) -> Tuple[int, int, str]:
        status = normalize_status(finding.get("final_status"))
        severity = normalize_severity(finding.get("severity"))
        fid = as_text(finding.get("finding_id"))
        return (
            STATUS_ORDER.get(status, 99),
            SEVERITY_ORDER.get(severity, 99),
            fid,
        )

    return sorted(findings, key=key)


def filter_reportable_findings(findings: Iterable[Dict[str, Any]], include_no_gap: bool = False) -> List[Dict[str, Any]]:
    out = []
    for finding in findings:
        status = normalize_status(finding.get("final_status"))
        if include_no_gap or status in REPORTABLE_STATUSES:
            out.append(finding)
    return out


def count_by(findings: Iterable[Dict[str, Any]], field: str, normalizer=None) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for finding in findings:
        value = finding.get(field)
        if normalizer:
            value = normalizer(value)
        else:
            value = as_text(value, "unknown")
        counter[value] += 1
    return dict(counter)


def citation_to_markdown(citation: Dict[str, Any], citation_type: str) -> str:
    if citation_type == "guideline":
        guideline_id = as_text(citation.get("guideline_id"), "Guideline")
        source_file = as_text(citation.get("source_file"), "unknown source")
        page = as_text(citation.get("page_number"), "?")
        section = as_text(citation.get("section"), "?")
        chunk_id = as_text(citation.get("chunk_id"), "?")
        return f"{guideline_id}, {source_file}, page {page}, section {section}, chunk {chunk_id}"

    if citation_type == "rule_evidence":
        file_name = as_text(citation.get("file_name"), "unknown file")
        page = as_text(citation.get("page_number"), "?")
        section = as_text(citation.get("section"), "?")
        value = as_text(citation.get("value"), "")
        chunk_id = as_text(citation.get("chunk_id"), "?")
        value_part = f", value: {value}" if value else ""
        return f"{file_name}, page {page}, section {section}, chunk {chunk_id}{value_part}"

    file_name = as_text(citation.get("file_name"), "unknown file")
    page = as_text(citation.get("page_number"), "?")
    section = as_text(citation.get("section"), "?")
    chunk_id = as_text(citation.get("chunk_id"), "?")
    return f"{file_name}, page {page}, section {section}, chunk {chunk_id}"


def citations_block(title: str, citations: List[Dict[str, Any]], citation_type: str) -> List[str]:
    lines = [f"**{title}:**"]
    if not citations:
        lines.append("- None recorded.")
        return lines
    for citation in citations:
        lines.append(f"- {md_escape(citation_to_markdown(citation, citation_type))}")
    return lines


def build_report_object(reconciled: Dict[str, Any], include_no_gap: bool = False, dedupe_unsupported_claims: bool = True) -> Dict[str, Any]:
    raw_findings = reconciled.get("final_findings", [])
    if not isinstance(raw_findings, list):
        raw_findings = []

    findings = filter_reportable_findings(raw_findings, include_no_gap=include_no_gap)
    findings = deduplicate_findings(findings, dedupe_unsupported_claims=dedupe_unsupported_claims)
    findings = sort_findings(findings)

    summary = {
        "generated_at": utc_now_iso(),
        "source_input": as_text(reconciled.get("input_file") or reconciled.get("source_input")),
        "raw_final_findings": len(raw_findings),
        "reportable_findings": len(findings),
        "status_counts": count_by(findings, "final_status", normalize_status),
        "severity_counts": count_by(findings, "severity", normalize_severity),
        "finding_type_counts": count_by(findings, "finding_type", normalize_finding_type),
        "high_severity_count": sum(1 for f in findings if normalize_severity(f.get("severity")) == "high"),
        "deduplicated_unsupported_claims": dedupe_unsupported_claims,
    }

    return {
        "report_type": "evidence_backed_gap_report",
        "project_title": PROJECT_TITLE,
        "disclaimer": DISCLAIMER,
        "source_reconciled_report": reconciled.get("run_type", "reconciled_gap_report"),
        "summary": summary,
        "final_findings": findings,
    }


def write_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = report.get("summary", {})
    findings = report.get("final_findings", [])

    lines: List[str] = []
    lines.append(f"# {PROJECT_TITLE} - Evidence-Backed Gap Report")
    lines.append("")
    lines.append(f"Generated at: `{summary.get('generated_at', utc_now_iso())}`")
    lines.append("")
    lines.append(f"> **Disclaimer:** {DISCLAIMER}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Reportable findings: **{summary.get('reportable_findings', 0)}**")
    lines.append(f"- High severity findings: **{summary.get('high_severity_count', 0)}**")
    lines.append("")

    lines.append("### Final Status Counts")
    lines.append("")
    lines.append("| Final Status | Count |")
    lines.append("|---|---:|")
    for status, count in sorted(summary.get("status_counts", {}).items(), key=lambda kv: STATUS_ORDER.get(kv[0], 99)):
        lines.append(f"| {md_escape(status_label(status))} | {count} |")
    lines.append("")

    lines.append("### Severity Counts")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|---|---:|")
    for severity, count in sorted(summary.get("severity_counts", {}).items(), key=lambda kv: SEVERITY_ORDER.get(kv[0], 99)):
        lines.append(f"| {md_escape(severity_label(severity))} | {count} |")
    lines.append("")

    lines.append("## Finding Overview")
    lines.append("")
    lines.append("| ID | Final Status | Severity | Type | Summary | Reviewer Action |")
    lines.append("|---|---|---|---|---|---|")
    for finding in findings:
        lines.append(
            "| "
            f"{md_escape(finding.get('finding_id'))} | "
            f"{md_escape(status_label(normalize_status(finding.get('final_status'))))} | "
            f"{md_escape(severity_label(normalize_severity(finding.get('severity'))))} | "
            f"{md_escape(normalize_finding_type(finding.get('finding_type')))} | "
            f"{md_escape(finding.get('finding_summary'))} | "
            f"{md_escape(finding.get('reviewer_action'))} |"
        )
    lines.append("")

    lines.append("## Detailed Findings")
    lines.append("")
    for idx, finding in enumerate(findings, start=1):
        fid = as_text(finding.get("finding_id"), f"Finding-{idx}")
        status = normalize_status(finding.get("final_status"))
        severity = normalize_severity(finding.get("severity"))
        ftype = normalize_finding_type(finding.get("finding_type"))

        lines.append(f"### {idx}. {fid} - {status_label(status)}")
        lines.append("")
        lines.append(f"- **Severity:** {severity_label(severity)}")
        lines.append(f"- **Finding type:** `{ftype}`")
        lines.append(f"- **Source:** `{as_text(finding.get('source'), 'unknown')}`")
        lines.append(f"- **Package ID:** `{as_text(finding.get('package_id'), 'N/A')}`")
        lines.append(f"- **Rule ID:** `{as_text(finding.get('rule_id'), 'N/A')}`")
        lines.append(f"- **LLM decision:** `{as_text(finding.get('llm_decision'), 'N/A')}`")
        lines.append(f"- **Rule status:** `{as_text(finding.get('rule_status'), 'N/A')}`")
        lines.append(f"- **Evidence status:** `{as_text(finding.get('evidence_status'), 'N/A')}`")
        lines.append("")
        lines.append(f"**Summary:** {as_text(finding.get('finding_summary'), 'No summary provided.')}")
        lines.append("")
        reasoning = as_text(finding.get("reasoning_summary"), "")
        if reasoning:
            lines.append(f"**Reasoning summary:** {reasoning}")
            lines.append("")
        lines.append(f"**Reviewer action:** {as_text(finding.get('reviewer_action'), 'Review this finding.')}")
        lines.append("")
        reconciliation_reason = as_text(finding.get("reconciliation_reason"), "")
        if reconciliation_reason:
            lines.append(f"**Reconciliation reason:** {reconciliation_reason}")
            lines.append("")

        grouped_ids = finding.get("grouped_finding_ids")
        if isinstance(grouped_ids, list) and grouped_ids:
            lines.append(f"**Grouped source findings:** {', '.join(md_escape(x) for x in grouped_ids)}")
            lines.append("")

        lines.extend(citations_block("Guideline citations", finding.get("guideline_citations", []), "guideline"))
        lines.append("")
        lines.extend(citations_block("Dossier citations", finding.get("dossier_citations", []), "dossier"))
        lines.append("")
        lines.extend(citations_block("Rule evidence", finding.get("rule_evidence", []), "rule_evidence"))
        lines.append("")
        lines.append("---")
        lines.append("")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_csv_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    findings = report.get("final_findings", [])

    fields = [
        "finding_id",
        "final_status",
        "severity",
        "finding_type",
        "finding_summary",
        "reviewer_action",
        "source",
        "package_id",
        "rule_id",
        "llm_decision",
        "rule_status",
        "evidence_status",
        "reconciliation_reason",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for finding in findings:
            writer.writerow({field: as_text(finding.get(field)) for field in fields})


def generate_reports(
    input_file: Path = DEFAULT_INPUT,
    markdown_output: Path = DEFAULT_MARKDOWN_OUTPUT,
    json_output: Path = DEFAULT_JSON_OUTPUT,
    csv_output: Path = DEFAULT_CSV_OUTPUT,
    include_no_gap: bool = False,
    dedupe_unsupported_claims: bool = True,
) -> Dict[str, Any]:
    reconciled = read_json(input_file)
    report = build_report_object(
        reconciled,
        include_no_gap=include_no_gap,
        dedupe_unsupported_claims=dedupe_unsupported_claims,
    )

    write_json(report, json_output)
    write_markdown_report(report, markdown_output)
    write_csv_report(report, csv_output)

    return report


def print_summary(report: Dict[str, Any], markdown_output: Path, json_output: Path, csv_output: Path) -> None:
    summary = report.get("summary", {})
    print("Report generation complete")
    print(f"Reportable findings: {summary.get('reportable_findings', 0)}")
    print("Status counts:")
    for status, count in sorted(summary.get("status_counts", {}).items(), key=lambda kv: STATUS_ORDER.get(kv[0], 99)):
        print(f"  {status}: {count}")
    print("Outputs:")
    print(f"  Markdown: {markdown_output}")
    print(f"  JSON:     {json_output}")
    print(f"  CSV:      {csv_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final reviewer-facing gap reports.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--include-no-gap", action="store_true", help="Include no_gap_detected findings in the report.")
    parser.add_argument(
        "--no-dedupe-unsupported-claims",
        action="store_true",
        help="Disable grouping of repeated unsupported-claim findings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_reports(
        input_file=args.input,
        markdown_output=args.markdown,
        json_output=args.json,
        csv_output=args.csv,
        include_no_gap=args.include_no_gap,
        dedupe_unsupported_claims=not args.no_dedupe_unsupported_claims,
    )
    print_summary(report, args.markdown, args.json, args.csv)


if __name__ == "__main__":
    main()
