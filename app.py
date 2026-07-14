"""
Streamlit reviewer dashboard for ReguLens-AI.

Run from the project root:
    streamlit run app.py

Requires the accompanying .streamlit/config.toml (dark theme) to sit in the
same project root -- Streamlit picks it up automatically.

This app can:
1. Accept dossier document uploads and show live ingestion stats.
2. Run the main analysis pipeline step-by-step.
3. Generate reviewer-facing reports.
4. Display final reconciled findings with filters, citations, and downloads.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import subprocess

import pandas as pd
import streamlit as st

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploaded_dossiers"

LLM_DECISIONS_PATH = OUTPUTS_DIR / "llm_decisions.json"
LLM_DECISIONS_V2_PATH = OUTPUTS_DIR / "llm_decisions_v2_d2g.json"
RULE_RESULTS_PATH = OUTPUTS_DIR / "rule_verification_results.json"
RECONCILED_REPORT_PATH = OUTPUTS_DIR / "reconciled_gap_report.json"
GAP_REPORT_JSON_PATH = OUTPUTS_DIR / "gap_report.json"
GAP_REPORT_MD_PATH = OUTPUTS_DIR / "gap_report.md"
GAP_REPORT_CSV_PATH = OUTPUTS_DIR / "gap_report.csv"

DEFAULT_MODEL_PATH = "models/qwen_regulatory_merged_v2"
ACCEPTED_UPLOAD_TYPES = ["pdf", "docx", "doc", "txt"]


# -----------------------------------------------------------------------------
# Streamlit config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="ReguLens-AI Reviewer Dashboard",
    page_icon="\U0001F9EA",  # test tube
    layout="wide",
)


# -----------------------------------------------------------------------------
# Visual identity
#
# The dark theme itself lives in .streamlit/config.toml -- that's the
# supported way to theme every native Streamlit widget (buttons, inputs,
# dataframes, file uploader, tabs) consistently, so nothing fights the
# user's browser/OS theme. This CSS block only adds the handful of things
# config.toml can't: the header banner, the disclaimer strip, stat cards,
# and status/severity badges.
# -----------------------------------------------------------------------------

APP_CSS = """
<style>
:root {
    --rl-panel: #17233A;
    --rl-border: #26334A;
    --rl-text-muted: #9AA7C2;
    --rl-amber: #D69A3B;
}

.rl-banner {
    background: linear-gradient(90deg, #101826 0%, #1B2A45 100%);
    border-left: 5px solid var(--rl-amber);
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 16px;
}
.rl-banner h1 {
    color: #FFFFFF;
    margin: 0 0 6px 0;
    font-size: 1.6rem;
}
.rl-banner p {
    color: var(--rl-text-muted);
    margin: 0;
    font-size: 0.92rem;
}

.rl-disclaimer {
    background: rgba(214, 154, 59, 0.12);
    border: 1px solid rgba(214, 154, 59, 0.4);
    border-left: 4px solid var(--rl-amber);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.85rem;
    color: #E8C888;
    margin-bottom: 20px;
}

.rl-stat-card {
    background: var(--rl-panel);
    border: 1px solid var(--rl-border);
    border-radius: 10px;
    padding: 16px 18px;
}
.rl-stat-card .rl-stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--rl-text-muted);
    margin-bottom: 4px;
}
.rl-stat-card .rl-stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFFFFF;
}

.badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-right: 6px;
}
.badge-red { background: #C0392B; color: #FFFFFF; }
.badge-orange { background: #C56A15; color: #FFFFFF; }
.badge-gold { background: #D4A017; color: #1B1B1B; }
.badge-blue { background: #3B5BDB; color: #FFFFFF; }
.badge-green { background: #2F9E58; color: #FFFFFF; }
.badge-gray { background: #4B5568; color: #E8EAF0; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border-color: var(--rl-border) !important;
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


BADGE_KIND_BY_SEVERITY = {
    "high": "red",
    "medium": "orange",
    "low": "gold",
    "none": "gray",
    "unknown": "gray",
}

BADGE_KIND_BY_STATUS = {
    "confirmed_gap": "red",
    "rule_flagged_gap": "orange",
    "potential_gap": "gold",
    "needs_human_review": "blue",
    "unconfirmed_llm_flag": "gray",
    "no_gap_detected": "green",
    "unknown": "gray",
}


def badge(text: str, kind: str) -> str:
    """Return a small HTML pill badge. Text is escaped defensively since it
    ultimately originates from pipeline-generated JSON, not hardcoded strings."""
    safe_text = html_escape(str(text) or "unknown")
    css_class = f"badge badge-{kind}"
    return f'<span class="{css_class}">{safe_text}</span>'


def stat_card(label: str, value: str) -> str:
    return (
        '<div class="rl-stat-card">'
        f'<div class="rl-stat-label">{html_escape(label)}</div>'
        f'<div class="rl-stat-value">{html_escape(value)}</div>'
        "</div>"
    )


def show_result(ok: bool, success_text: str, error_text: str) -> None:
    """
    Explicit if/else result banner. NOTE: this deliberately avoids the
    `st.success(x) if ok else st.error(y)` ternary-as-statement pattern --
    that bare expression gets picked up by Streamlit's "magic" auto-display
    (it isn't excluded the way a plain function-call statement is), which
    prints the raw DeltaGenerator repr into the app. Always call st.success/
    st.error from inside a real if/else block instead.
    """
    if ok:
        st.success(success_text)
    else:
        st.error(error_text)


PDF_SEVERITY_RGB = {
    "high": (192, 57, 43),
    "medium": (197, 106, 21),
    "low": (212, 160, 23),
    "none": (130, 130, 130),
    "unknown": (130, 130, 130),
}


def _pdf_safe(value: Any) -> str:
    """PDF core fonts only support latin-1, so replace anything outside
    that range rather than letting fpdf raise on an unusual character."""
    text = "" if value is None else str(value)
    return text.encode("latin-1", "replace").decode("latin-1")


def _pdf_line(pdf: "FPDF", text: str, size: float = 10, bold: bool = False,
              color: Tuple[int, int, int] = (20, 20, 20), h: float = 5) -> None:
    """
    Write one paragraph, always starting from the left margin with the
    full content width explicitly computed. Some fpdf2 versions don't
    reliably reset the cursor to the left margin after multi_cell(), which
    can leave almost no horizontal room for the next line and raise
    'Not enough horizontal space to render a single character'. Resetting
    the x position ourselves before every call avoids that entirely.
    """
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.set_text_color(*color)
    content_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(content_width, h, _pdf_safe(text))


def _pdf_divider(pdf: "FPDF") -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.2)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(3)


@st.cache_data(show_spinner=False)
def build_pdf_report(findings: List[Dict[str, Any]], summary: Dict[str, Any]) -> bytes:
    """Render the findings into a standalone PDF reviewers can open without
    any other tooling. Cached on (findings, summary) so retyping a filter
    or search box elsewhere in the app doesn't regenerate it needlessly."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    _pdf_line(pdf, "ReguLens-AI: Regulatory Dossier Gap Report", size=16, bold=True, h=9)
    _pdf_line(pdf, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", size=9, color=(140, 140, 140))
    pdf.ln(2)

    _pdf_line(
        pdf,
        "Reviewer-assistance prototype only. This report does not certify "
        "regulatory compliance, approve, reject, or replace human regulatory judgment.",
        size=9, color=(60, 60, 60),
    )
    pdf.ln(4)

    status_counter = Counter(normalize_text(x.get("final_status")) or "unknown" for x in findings)
    severity_counter = Counter(normalize_text(x.get("severity")) or "unknown" for x in findings)
    metrics = [
        ("Total findings", str(len(findings))),
        ("Confirmed gaps", str(status_counter.get("confirmed_gap", 0))),
        ("Rule-flagged", str(status_counter.get("rule_flagged_gap", 0))),
        ("Potential gaps", str(status_counter.get("potential_gap", 0))),
        ("High severity", str(severity_counter.get("high", 0))),
    ]
    _pdf_line(pdf, "Summary", size=11, bold=True, h=6)
    for label, value in metrics:
        _pdf_line(pdf, f"{label}: {value}", size=10, h=6)
    pdf.ln(4)

    ordered = sorted(
        findings,
        key=lambda x: (
            status_rank(normalize_text(x.get("final_status"))),
            severity_rank(normalize_text(x.get("severity"))),
            normalize_text(x.get("finding_id")),
        ),
    )

    for finding in ordered:
        finding_id = normalize_text(finding.get("finding_id")) or "Finding"
        final_status = normalize_text(finding.get("final_status")) or "unknown"
        severity = normalize_text(finding.get("severity")) or "unknown"
        finding_type = normalize_text(finding.get("finding_type")) or "unknown"
        summary_text = normalize_text(finding.get("finding_summary")) or "No summary available."
        rgb = PDF_SEVERITY_RGB.get(severity.lower(), (130, 130, 130))

        _pdf_divider(pdf)

        _pdf_line(pdf, finding_id, size=12, bold=True, h=6)
        _pdf_line(
            pdf,
            f"{severity.upper()}  |  {final_status.replace('_', ' ').upper()}  |  {finding_type}",
            size=9, bold=True, color=rgb, h=5,
        )
        _pdf_line(pdf, summary_text, size=10, h=5)
        pdf.ln(1)

        reviewer_action = normalize_text(finding.get("reviewer_action"))
        if reviewer_action:
            _pdf_line(pdf, "Reviewer Action", size=9, bold=True, h=5)
            _pdf_line(pdf, reviewer_action, size=9, h=5)

        reasoning = normalize_text(finding.get("reasoning_summary"))
        if reasoning:
            _pdf_line(pdf, "Reasoning Summary", size=9, bold=True, h=5)
            _pdf_line(pdf, reasoning, size=9, h=5)

        reconciliation_reason = normalize_text(finding.get("reconciliation_reason"))
        if reconciliation_reason:
            _pdf_line(pdf, "Reconciliation Reason", size=9, bold=True, h=5)
            _pdf_line(pdf, reconciliation_reason, size=9, h=5)

        guideline_citations = finding.get("guideline_citations")
        if isinstance(guideline_citations, list) and guideline_citations:
            _pdf_line(pdf, "Guideline Citations", size=9, bold=True, h=5)
            for cite in guideline_citations:
                _pdf_line(pdf, f"- {format_citation(cite)}", size=9, h=5)

        dossier_citations = finding.get("dossier_citations")
        if isinstance(dossier_citations, list) and dossier_citations:
            _pdf_line(pdf, "Dossier Citations", size=9, bold=True, h=5)
            for cite in dossier_citations:
                _pdf_line(pdf, f"- {format_citation(cite)}", size=9, h=5)

        pdf.ln(3)

    raw = pdf.output(dest="S")
    return bytes(raw) if not isinstance(raw, str) else raw.encode("latin-1")


# -----------------------------------------------------------------------------
# Utility helpers (pipeline / report)
# -----------------------------------------------------------------------------


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def load_json(path: Path) -> Optional[Any]:
    if not file_exists(path):
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - UI display only
        st.error(f"Could not read {path}: {exc}")
        return None


def read_text(path: Path) -> str:
    if not file_exists(path):
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def run_command(command: List[str], label: str) -> Tuple[bool, str]:
    """Run a project command and return success + combined output."""
    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return False, f"Command failed: {exc}"

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += "\n" + result.stderr

    if result.returncode != 0:
        return False, output.strip() or f"{label} failed with exit code {result.returncode}."

    return True, output.strip() or f"{label} completed successfully."


def copy_file(src: Path, dst: Path) -> None:
    if file_exists(src):
        dst.write_bytes(src.read_bytes())


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_findings(report: Any) -> List[Dict[str, Any]]:
    """Accept either report_generator output or reconciler output."""
    if isinstance(report, list):
        return [x for x in report if isinstance(x, dict)]

    if not isinstance(report, dict):
        return []

    for key in [
        "findings",
        "final_findings",
        "report_findings",
        "gap_findings",
        "items",
    ]:
        value = report.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    return []


def get_summary(report: Any, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(report, dict) and isinstance(report.get("summary"), dict):
        return report["summary"]

    return {
        "total_findings": len(findings),
        "by_final_status": dict(Counter(normalize_text(x.get("final_status")) or "unknown" for x in findings)),
        "by_severity": dict(Counter(normalize_text(x.get("severity")) or "unknown" for x in findings)),
        "by_finding_type": dict(Counter(normalize_text(x.get("finding_type")) or "unknown" for x in findings)),
    }


def unique_values(findings: Iterable[Dict[str, Any]], field: str) -> List[str]:
    values = sorted({normalize_text(x.get(field)) for x in findings if normalize_text(x.get(field))})
    return values


def severity_rank(severity: str) -> int:
    order = {
        "high": 0,
        "medium": 1,
        "low": 2,
        "none": 3,
        "unknown": 4,
    }
    return order.get(normalize_text(severity).lower(), 4)


def status_rank(status: str) -> int:
    order = {
        "confirmed_gap": 0,
        "rule_flagged_gap": 1,
        "potential_gap": 2,
        "needs_human_review": 3,
        "unconfirmed_llm_flag": 4,
        "no_gap_detected": 5,
    }
    return order.get(normalize_text(status).lower(), 9)


def format_citation(citation: Dict[str, Any]) -> str:
    if not isinstance(citation, dict):
        return str(citation)

    parts = []
    for key in ["guideline_id", "source_file", "file_name", "page_number", "section", "chunk_id"]:
        value = citation.get(key)
        if value is not None and str(value).strip():
            label = key.replace("_", " ").title()
            parts.append(f"**{label}:** {value}")
    return "  \n".join(parts) if parts else json.dumps(citation, ensure_ascii=False)


def render_citations(title: str, citations: Any) -> None:
    if not isinstance(citations, list) or not citations:
        st.caption(f"No {title.lower()} available.")
        return

    for idx, citation in enumerate(citations, start=1):
        st.markdown(f"**{title} {idx}**")
        st.markdown(format_citation(citation))


def render_rule_evidence(evidence: Any) -> None:
    if not isinstance(evidence, list) or not evidence:
        st.caption("No rule evidence available.")
        return

    for idx, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            st.write(item)
            continue
        value = item.get("value", "")
        file_name = item.get("file_name", "")
        page = item.get("page_number", "")
        section = item.get("section", "")
        snippet = item.get("text_snippet", "")
        st.markdown(f"**Evidence {idx}:** `{value}`")
        st.caption(f"{file_name} | page {page} | section {section}")
        if snippet:
            st.write(snippet)


def render_finding_card(finding: Dict[str, Any]) -> None:
    finding_id = normalize_text(finding.get("finding_id")) or "Finding"
    final_status = normalize_text(finding.get("final_status")) or "unknown"
    severity = normalize_text(finding.get("severity")) or "unknown"
    finding_type = normalize_text(finding.get("finding_type")) or "unknown"
    summary = normalize_text(finding.get("finding_summary")) or "No summary available."

    status_kind = BADGE_KIND_BY_STATUS.get(final_status.lower(), "gray")
    severity_kind = BADGE_KIND_BY_SEVERITY.get(severity.lower(), "gray")

    with st.container(border=True):
        top_left, top_right = st.columns([4, 2])
        with top_left:
            st.markdown(f"#### {html_escape(finding_id)}")
            st.markdown(html_escape(summary))
        with top_right:
            badges_html = (
                badge(severity, severity_kind)
                + badge(final_status.replace("_", " "), status_kind)
                + badge(finding_type, "gray")
            )
            st.markdown(f'<div style="text-align:right">{badges_html}</div>', unsafe_allow_html=True)

        st.markdown("**Reviewer Action**")
        st.write(normalize_text(finding.get("reviewer_action")) or "Review finding and resolve the dossier issue.")

        with st.expander("Reasoning and reconciliation details"):
            col1, col2, col3 = st.columns(3)
            col1.metric("LLM decision", normalize_text(finding.get("llm_decision")) or "n/a")
            col2.metric("Rule status", normalize_text(finding.get("rule_status")) or "n/a")
            col3.metric("Evidence status", normalize_text(finding.get("evidence_status")) or "n/a")

            if finding.get("rule_id"):
                st.markdown(f"**Rule ID:** `{finding.get('rule_id')}`")

            reasoning = normalize_text(finding.get("reasoning_summary"))
            if reasoning:
                st.markdown("**LLM reasoning summary**")
                st.write(reasoning)

            reconciliation_reason = normalize_text(finding.get("reconciliation_reason"))
            if reconciliation_reason:
                st.markdown("**Reconciliation reason**")
                st.write(reconciliation_reason)

        with st.expander("Citations and evidence"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Guideline citations")
                render_citations("Guideline citation", finding.get("guideline_citations"))
            with c2:
                st.markdown("##### Dossier citations")
                render_citations("Dossier citation", finding.get("dossier_citations"))

            st.markdown("##### Rule evidence")
            render_rule_evidence(finding.get("rule_evidence"))


# -----------------------------------------------------------------------------
# Upload / ingestion helpers
# -----------------------------------------------------------------------------


def ensure_upload_dir() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def human_readable_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def list_uploaded_documents() -> List[Dict[str, Any]]:
    ensure_upload_dir()
    docs = []
    for p in UPLOADS_DIR.iterdir():
        if p.is_file():
            stat = p.stat()
            docs.append(
                {
                    "File name": p.name,
                    "Type": (p.suffix.lower().lstrip(".") or "unknown"),
                    "Size": human_readable_size(stat.st_size),
                    "_size_bytes": stat.st_size,
                    "Uploaded": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "_mtime": stat.st_mtime,
                }
            )
    docs.sort(key=lambda d: d["_mtime"], reverse=True)
    return docs


def save_uploaded_files(uploaded_files: List[Any]) -> Tuple[int, int]:
    """
    Persist newly-uploaded files to UPLOADS_DIR. Skips files already saved
    with an identical name and size (so re-running the app with the same
    selection doesn't double-count). Shows a progress bar for reviewer
    visibility during the save. Returns (files_newly_saved, bytes_newly_saved).
    """
    ensure_upload_dir()
    if not uploaded_files:
        return 0, 0

    existing = {(p.name, p.stat().st_size) for p in UPLOADS_DIR.iterdir() if p.is_file()}

    newly_saved = 0
    bytes_saved = 0
    progress = st.progress(0, text="Saving uploaded documents...")
    total = len(uploaded_files)

    for i, uf in enumerate(uploaded_files, start=1):
        data = uf.getvalue()
        key = (uf.name, len(data))
        if key not in existing:
            dest = UPLOADS_DIR / uf.name
            dest.write_bytes(data)
            newly_saved += 1
            bytes_saved += len(data)
        progress.progress(i / total, text=f"Saving uploaded documents... ({i}/{total})")

    progress.empty()
    return newly_saved, bytes_saved


def clear_uploaded_documents() -> int:
    ensure_upload_dir()
    removed = 0
    for p in UPLOADS_DIR.iterdir():
        if p.is_file():
            p.unlink()
            removed += 1
    return removed



# -----------------------------------------------------------------------------
# Page flow (session-state driven, since Streamlit has no built-in navigation)
#
# Three screens, one at a time:
#   1. landing  - full-bleed visual intro; click anywhere to continue
#   2. upload   - upload dossiers, tweak advanced settings if needed, run
#   3. report   - summary, downloads, and the filterable findings list
# -----------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "landing"


def go_to(page_name: str) -> None:
    st.session_state.page = page_name


# Hide Streamlit's default chrome (hamburger menu / footer) everywhere so the
# three screens feel like a single focused app rather than a generic
# Streamlit page. Safe no-op if these testids ever change in a future
# Streamlit version -- the app just falls back to showing the default chrome.
st.markdown(
    """
    <style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Screen 1: Landing
# -----------------------------------------------------------------------------

LANDING_CSS = """
<style>
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.landing-hero {
    position: relative;
    min-height: 92vh;
    overflow: hidden;
    background: radial-gradient(circle at 15% 20%, #1B2A45 0%, #0B111D 55%, #070A11 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 64px 72px;
}
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(46px);
    opacity: 0.55;
    mix-blend-mode: screen;
    pointer-events: none;
}
.orb-a { width: 360px; height: 360px; top: 6%; right: 14%;
    background: radial-gradient(circle at 32% 32%, #FFC98B, #7A3B12); }
.orb-b { width: 240px; height: 240px; top: 38%; right: 34%;
    background: radial-gradient(circle at 32% 32%, #8C9CFF, #1C2450); }
.orb-c { width: 170px; height: 170px; bottom: 10%; right: 8%;
    background: radial-gradient(circle at 32% 32%, #FFC98B, #7A3B12); }
.orb-d { width: 130px; height: 130px; top: 58%; right: 6%;
    background: radial-gradient(circle at 32% 32%, #8C9CFF, #1C2450); }

.landing-brand {
    position: absolute; top: 36px; left: 44px; z-index: 3;
    color: #FFFFFF; font-weight: 800; font-size: 1.7rem; letter-spacing: 0.03em;
    text-shadow: 0 0 24px rgba(214, 154, 59, 0.35);
}
.landing-brand .accent { color: #D69A3B; }
.landing-title {
    color: #FFFFFF; font-weight: 800; font-size: 4rem; line-height: 1.05;
    letter-spacing: -0.02em; max-width: 720px; z-index: 2;
}
.landing-title .accent { color: #D69A3B; }
.landing-sub {
    color: #9AA7C2; font-size: 1.05rem; max-width: 520px; margin-top: 20px; z-index: 2;
}
.landing-hint {
    position: absolute; bottom: 34px; left: 44px;
    color: #7D8BAB; font-size: 0.85rem; letter-spacing: 0.02em;
}

/* Make the (invisible) Streamlit button behind this cover the full
   viewport, so clicking anywhere on the hero advances to the next screen.
   This relies on the landing screen having exactly one button rendered. */
div[data-testid="stButton"] {
    position: fixed; inset: 0; z-index: 999; margin: 0;
}
div[data-testid="stButton"] button {
    width: 100vw; height: 100vh; opacity: 0; cursor: pointer; border: none;
}
</style>
"""


def render_landing_page() -> None:
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="landing-hero">
            <div class="orb orb-a"></div>
            <div class="orb orb-b"></div>
            <div class="orb orb-c"></div>
            <div class="orb orb-d"></div>
            <div class="landing-brand">ReguLens<span class="accent">-AI</span></div>
            <div class="landing-title">Find the&mdash;<br><span class="accent">unexpected</span> gaps.</div>
            <div class="landing-sub">
                Every CTD dossier, checked against every guideline it has to satisfy
                &mdash; before a reviewer ever has to ask why.
            </div>
            <div class="landing-hint">Click anywhere to continue &rarr;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Enter", key="landing_enter_button"):
        go_to("upload")
        st.rerun()


# -----------------------------------------------------------------------------
# Screen 2: Upload & Run
# -----------------------------------------------------------------------------

def render_upload_page() -> None:
    st.markdown(
        """
        <div class="rl-banner">
            <h1>ReguLens-AI: Regulatory Dossier Gap Review</h1>
            <p>Upload the dossier documents you need checked, then run the analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rl-disclaimer">Reviewer-assistance prototype only. This tool does not certify '
        "regulatory compliance, approve, reject, or replace human regulatory judgment.</div>",
        unsafe_allow_html=True,
    )

    existing_report = load_json(GAP_REPORT_JSON_PATH) or load_json(RECONCILED_REPORT_PATH)
    if existing_report is not None:
        top_l, top_r = st.columns([4, 1])
        with top_l:
            st.caption("A report from a previous run is already available.")
        with top_r:
            if st.button("View last report", width="stretch"):
                go_to("report")
                st.rerun()

    st.markdown("### 1. Upload documents")

    uploaded_files = st.file_uploader(
        "Upload dossier documents (PDF, DOCX, DOC, TXT)",
        type=ACCEPTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Files are saved locally to data/uploaded_dossiers/. Nothing leaves this machine.",
    )

    if uploaded_files:
        newly_saved, bytes_saved = save_uploaded_files(uploaded_files)
        if newly_saved:
            st.success(f"Saved {newly_saved} new document(s) ({human_readable_size(bytes_saved)}).")

    documents = list_uploaded_documents()
    total_docs = len(documents)
    total_bytes = sum(d["_size_bytes"] for d in documents)
    last_upload = documents[0]["Uploaded"] if documents else "\u2014"

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(stat_card("Documents uploaded", str(total_docs)), unsafe_allow_html=True)
    with s2:
        st.markdown(stat_card("Total size ingested", human_readable_size(total_bytes)), unsafe_allow_html=True)
    with s3:
        st.markdown(stat_card("Last upload", last_upload), unsafe_allow_html=True)

    if documents:
        with st.expander(f"View {total_docs} uploaded document(s)"):
            display_df = pd.DataFrame(
                [{"File name": d["File name"], "Type": d["Type"], "Size": d["Size"], "Uploaded": d["Uploaded"]} for d in documents]
            )
            st.dataframe(display_df, width="stretch", hide_index=True)

            confirm_clear = st.checkbox(f"I understand this will permanently delete all {total_docs} document(s).")
            if st.button("Clear uploaded documents", disabled=not confirm_clear):
                removed = clear_uploaded_documents()
                st.success(f"Removed {removed} document(s).")
                st.rerun()
    else:
        st.info("No documents uploaded yet. Use the uploader above to add dossier files.")

    st.markdown("### 2. Run analysis")
    st.caption("One click: builds evidence, runs the model, checks rules, and prepares the report.")

    with st.expander("\u2699\ufe0f Advanced settings (developer)", expanded=False):
        st.caption("Everyday reviewers shouldn't need anything in here.")
        model_path = st.text_input("LLM model path", DEFAULT_MODEL_PATH)
        max_packages = st.number_input("Max dossier-to-guideline packages", min_value=1, max_value=200, value=28)
        input_dir_override = st.text_input(
            "Dossier input directory (passed to evidence_builder)",
            value=str(UPLOADS_DIR.relative_to(PROJECT_ROOT)),
            help="Only change this if evidence_builder expects a different folder.",
        )

        st.markdown("##### Run individual steps")
        st.caption("Use these only to debug a single stage.")

        if st.button("1. Run LLM decision layer", width="stretch"):
            command = [
                sys.executable, "-m", "src.llm_decision_layer",
                "--model", model_path,
                "--direction", "dossier_to_guideline",
                "--max-packages", str(int(max_packages)),
            ]
            with st.spinner("Running LLM decision layer..."):
                ok, output = run_command(command, "LLM decision layer")
            if ok:
                copy_file(LLM_DECISIONS_PATH, LLM_DECISIONS_V2_PATH)
            show_result(ok, "LLM decisions created.", "LLM step failed.")
            with st.expander("LLM command output", expanded=not ok):
                st.code(output or "No output")

        if st.button("2. Run rule verifier", width="stretch"):
            command = [sys.executable, "-m", "src.rule_verifier"]
            with st.spinner("Running rule verifier..."):
                ok, output = run_command(command, "Rule verifier")
            show_result(ok, "Rule verification completed.", "Rule verifier failed.")
            with st.expander("Rule verifier output", expanded=not ok):
                st.code(output or "No output")

        if st.button("3. Run reconciler", width="stretch"):
            llm_path = LLM_DECISIONS_V2_PATH if file_exists(LLM_DECISIONS_V2_PATH) else LLM_DECISIONS_PATH
            command = [
                sys.executable, "-m", "src.reconciler",
                "--llm-decisions", str(llm_path.relative_to(PROJECT_ROOT)),
                "--rule-results", str(RULE_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            ]
            with st.spinner("Running reconciler..."):
                ok, output = run_command(command, "Reconciler")
            show_result(ok, "Reconciliation completed.", "Reconciler failed.")
            with st.expander("Reconciler output", expanded=not ok):
                st.code(output or "No output")

        if st.button("4. Generate final report", width="stretch"):
            command = [sys.executable, "-m", "src.report_generator"]
            with st.spinner("Generating report..."):
                ok, output = run_command(command, "Report generator")
            show_result(ok, "Report generated.", "Report generation failed.")
            with st.expander("Report generator output", expanded=not ok):
                st.code(output or "No output")

    if st.button(
        "\U0001F680 Run Full Analysis",
        type="primary",
        width="stretch",
        disabled=total_docs == 0,
    ):
        steps = [
            (
                "Evidence builder",
                [sys.executable, "-m", "src.evidence_builder", "--input-dir", input_dir_override],
            ),
            (
                "LLM decision layer",
                [
                    sys.executable, "-m", "src.llm_decision_layer",
                    "--model", model_path,
                    "--direction", "dossier_to_guideline",
                    "--max-packages", str(int(max_packages)),
                ],
            ),
            ("Rule verifier", [sys.executable, "-m", "src.rule_verifier"]),
            (
                "Reconciler",
                [
                    sys.executable, "-m", "src.reconciler",
                    "--llm-decisions",
                    str((LLM_DECISIONS_V2_PATH if file_exists(LLM_DECISIONS_V2_PATH) else LLM_DECISIONS_PATH).relative_to(PROJECT_ROOT)),
                    "--rule-results", str(RULE_RESULTS_PATH.relative_to(PROJECT_ROOT)),
                ],
            ),
            ("Report generator", [sys.executable, "-m", "src.report_generator"]),
        ]

        all_ok = True
        logs = []
        progress = st.progress(0, text="Starting analysis...")
        with st.spinner("Running full analysis on uploaded documents..."):
            for i, (label, command) in enumerate(steps, start=1):
                progress.progress((i - 1) / len(steps), text=f"Running: {label}")
                ok, output = run_command(command, label)
                logs.append(f"### {label}\n{output}")
                if label == "LLM decision layer" and ok:
                    copy_file(LLM_DECISIONS_PATH, LLM_DECISIONS_V2_PATH)
                if not ok:
                    all_ok = False
                    break
            progress.progress(1.0, text="Done")
        progress.empty()

        if all_ok:
            st.success("Analysis complete. Opening the report...")
            go_to("report")
            st.rerun()
        else:
            st.error("Something failed partway through \u2014 see the log below.")
            with st.expander("Run log", expanded=True):
                st.code("\n\n".join(logs))

    st.caption("Runs entirely locally. No internet access or external uploads.")


# -----------------------------------------------------------------------------
# Screen 3: Report
# -----------------------------------------------------------------------------

def render_report_page() -> None:
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.markdown(
            """
            <div class="rl-banner">
                <h1>ReguLens-AI: Gap Report</h1>
                <p>Review the findings below and download the report in whichever format you need.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_r:
        st.write("")
        if st.button("\u2190 Run another analysis", width="stretch"):
            go_to("upload")
            st.rerun()

    st.markdown(
        '<div class="rl-disclaimer">Reviewer-assistance prototype only. This tool does not certify '
        "regulatory compliance, approve, reject, or replace human regulatory judgment.</div>",
        unsafe_allow_html=True,
    )

    report = load_json(GAP_REPORT_JSON_PATH)
    report_source = GAP_REPORT_JSON_PATH
    if report is None:
        report = load_json(RECONCILED_REPORT_PATH)
        report_source = RECONCILED_REPORT_PATH

    if report is None:
        st.warning("No report found yet. Go back and run the analysis first.")
        if st.button("\u2190 Go to Upload & Run"):
            go_to("upload")
            st.rerun()
        return

    findings = get_findings(report)
    summary = get_summary(report, findings)

    st.caption(f"Loaded report: `{report_source.relative_to(PROJECT_ROOT)}`")

    status_counter = Counter(normalize_text(x.get("final_status")) or "unknown" for x in findings)
    severity_counter = Counter(normalize_text(x.get("severity")) or "unknown" for x in findings)

    total = len(findings)
    confirmed = status_counter.get("confirmed_gap", 0)
    rule_flagged = status_counter.get("rule_flagged_gap", 0)
    potential = status_counter.get("potential_gap", 0)
    high = severity_counter.get("high", 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(stat_card("Total findings", str(total)), unsafe_allow_html=True)
    with m2:
        st.markdown(stat_card("Confirmed gaps", str(confirmed)), unsafe_allow_html=True)
    with m3:
        st.markdown(stat_card("Rule-flagged", str(rule_flagged)), unsafe_allow_html=True)
    with m4:
        st.markdown(stat_card("Potential gaps", str(potential)), unsafe_allow_html=True)
    with m5:
        st.markdown(stat_card("High severity", str(high)), unsafe_allow_html=True)

    st.markdown("### Download the report")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        if file_exists(GAP_REPORT_MD_PATH):
            st.download_button(
                "Markdown", data=read_text(GAP_REPORT_MD_PATH),
                file_name="gap_report.md", mime="text/markdown", width="stretch",
            )
    with d2:
        if file_exists(GAP_REPORT_CSV_PATH):
            st.download_button(
                "CSV", data=GAP_REPORT_CSV_PATH.read_bytes(),
                file_name="gap_report.csv", mime="text/csv", width="stretch",
            )
    with d3:
        if file_exists(GAP_REPORT_JSON_PATH):
            st.download_button(
                "JSON", data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name="gap_report.json", mime="application/json", width="stretch",
            )
    with d4:
        if FPDF_AVAILABLE:
            try:
                pdf_bytes = build_pdf_report(findings, summary)
                st.download_button(
                    "PDF", data=pdf_bytes,
                    file_name="gap_report.pdf", mime="application/pdf", width="stretch",
                )
            except Exception as exc:
                st.caption(f"PDF export failed ({exc}). Use another format above.")
        else:
            st.caption("PDF needs `pip install fpdf2`")

    st.markdown("---")
    st.markdown("## Findings")

    statuses = unique_values(findings, "final_status")
    severities = unique_values(findings, "severity")
    finding_types = unique_values(findings, "finding_type")

    f1, f2, f3 = st.columns(3)
    selected_statuses = f1.multiselect("Final status", statuses, default=statuses)
    selected_severities = f2.multiselect("Severity", severities, default=severities)
    selected_types = f3.multiselect("Finding type", finding_types, default=finding_types)

    search_text = st.text_input("Search findings", placeholder="Search summary, reviewer action, citations...")

    filtered = []
    for finding in findings:
        if selected_statuses and normalize_text(finding.get("final_status")) not in selected_statuses:
            continue
        if selected_severities and normalize_text(finding.get("severity")) not in selected_severities:
            continue
        if selected_types and normalize_text(finding.get("finding_type")) not in selected_types:
            continue
        if search_text:
            blob = json.dumps(finding, ensure_ascii=False).lower()
            if search_text.lower() not in blob:
                continue
        filtered.append(finding)

    filtered.sort(
        key=lambda x: (
            status_rank(normalize_text(x.get("final_status"))),
            severity_rank(normalize_text(x.get("severity"))),
            normalize_text(x.get("finding_id")),
        )
    )

    st.caption(f"Showing {len(filtered)} of {len(findings)} findings.")

    for finding in filtered:
        render_finding_card(finding)


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

if st.session_state.page == "landing":
    render_landing_page()
elif st.session_state.page == "upload":
    render_upload_page()
elif st.session_state.page == "report":
    render_report_page()
else:
    st.session_state.page = "landing"
    st.rerun()