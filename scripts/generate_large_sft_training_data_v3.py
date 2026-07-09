"""
scripts/generate_large_sft_training_data_v3.py

Synthetic SFT dataset generator for the regulatory dossier gap assistant
LLM decision layer.

This version uses the exact assistant-output schema expected by
src/llm_decision_layer.py:

    llm_decision: gap | no_gap | uncertain | needs_human_review
    finding_type: missing_section | value_inconsistency | entity_inconsistency |
                  endpoint_inconsistency | safety_count_inconsistency |
                  unsupported_claim | weak_justification |
                  insufficient_evidence | no_gap | other
    severity: high | medium | low | none

Outputs:
    training_data/v3/llm_sft_train.jsonl
    training_data/v3/llm_sft_eval.jsonl
    training_data/v3/llm_sft_test.jsonl
    training_data/v3/dataset_manifest.json
    training_data/v3/error_analysis_targets.json

Run from project root:
    python -m scripts.generate_large_sft_training_data_v3
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


# =============================================================================
# 1. CONSTANTS
# =============================================================================

DATASET_VERSION = "v3"
SEED = 42
OUTPUT_DIR = Path("training_data/v3")

TRAIN_PATH = OUTPUT_DIR / "llm_sft_train.jsonl"
EVAL_PATH = OUTPUT_DIR / "llm_sft_eval.jsonl"
TEST_PATH = OUTPUT_DIR / "llm_sft_test.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.json"
ERROR_TARGETS_PATH = OUTPUT_DIR / "error_analysis_targets.json"

SYSTEM_PROMPT = """You are a regulatory dossier review assistant.

You will receive one neutral evidence package from a bidirectional Dual-RAG system.

Your task:
- Compare guideline expectations with retrieved dossier evidence.
- Decide whether the evidence indicates a regulatory gap, no gap, uncertainty, or need for human review.
- Use only the provided evidence.
- Do not use outside knowledge.
- Do not assume missing information unless the retrieved evidence is weak, absent, or only shows nearby unrelated content.
- Do not mention gold labels, expected answers, or test data.
- Return only valid JSON.
- Do not wrap the JSON in markdown.
- Do not include chain-of-thought. Provide only a concise reasoning_summary.

Allowed llm_decision values:
gap, no_gap, uncertain, needs_human_review

Allowed severity values:
high, medium, low, none

Allowed finding_type values:
missing_section, value_inconsistency, entity_inconsistency, endpoint_inconsistency, safety_count_inconsistency, unsupported_claim, weak_justification, insufficient_evidence, no_gap, other

Decision consistency rules:
- If finding_type is missing_section, value_inconsistency, entity_inconsistency, endpoint_inconsistency, safety_count_inconsistency, unsupported_claim, or weak_justification, then llm_decision should normally be "gap".
- If llm_decision is "no_gap", then finding_type must be "no_gap" and severity must be "none".
- If reasoning_summary describes a mismatch, contradiction, inconsistency, missing required section, unsupported claim, or weak justification, llm_decision must not be "no_gap".
- Deterministic mismatches such as shelf-life values, storage conditions, manufacturer names, batch identifiers, endpoint names, and serious adverse event counts should set needs_rule_verification to true.
- reviewer_action must never be blank.
- Guideline citations must cite only guideline evidence.
- Dossier citations must cite only dossier evidence.
"""

ALLOWED_DECISIONS = {"gap", "no_gap", "uncertain", "needs_human_review"}
ALLOWED_SEVERITIES = {"high", "medium", "low", "none"}
ALLOWED_EVIDENCE_STATUS = {"strong", "medium", "weak", "missing"}
ALLOWED_FINDING_TYPES = {
    "missing_section",
    "value_inconsistency",
    "entity_inconsistency",
    "endpoint_inconsistency",
    "safety_count_inconsistency",
    "unsupported_claim",
    "weak_justification",
    "insufficient_evidence",
    "no_gap",
    "other",
}
GAP_FINDING_TYPES = {
    "missing_section",
    "value_inconsistency",
    "entity_inconsistency",
    "endpoint_inconsistency",
    "safety_count_inconsistency",
    "unsupported_claim",
    "weak_justification",
}

ASSISTANT_KEYS = [
    "llm_decision",
    "finding_type",
    "severity",
    "evidence_status",
    "ctd_section",
    "finding_summary",
    "reasoning_summary",
    "guideline_citations",
    "dossier_citations",
    "reviewer_action",
    "needs_rule_verification",
]

ISSUE_FAMILIES = {
    "missing_section",
    "shelf_life_mismatch",
    "storage_condition_mismatch",
    "manufacturer_mismatch",
    "batch_mismatch",
    "endpoint_mismatch",
    "sae_count_mismatch",
    "unsupported_claim",
    "weak_justification",
    "no_gap_quality",
    "no_gap_clinical",
    "uncertain_ambiguous_evidence",
    "insufficient_evidence",
}

# Exactly 1000 examples. Distribution is intentionally close to:
# gap 56%, no_gap 25%, uncertain 10%, needs_human_review 9%.
FAMILY_COUNTS = {
    "missing_section": 70,
    "shelf_life_mismatch": 70,
    "storage_condition_mismatch": 70,
    "manufacturer_mismatch": 50,
    "batch_mismatch": 50,
    "endpoint_mismatch": 60,
    "sae_count_mismatch": 60,
    "unsupported_claim": 80,
    "weak_justification": 50,
    "no_gap_quality": 130,
    "no_gap_clinical": 120,
    "uncertain_ambiguous_evidence": 100,
    "insufficient_evidence": 90,
}

GUIDELINE_SOURCE_FILES = {
    "ICH_M4": "M4_R4__Guideline.pdf",
    "ICH_M4Q": "M4Q_R1_Guideline.pdf",
    "ICH_M4S": "M4S_R2_Guideline.pdf",
    "ICH_M4E": "M4E_R2__Guideline.pdf",
    "ICH_E3": "E3_Guideline.pdf",
    "ICH_Q1A": "Q1A(R2)_Guideline.pdf",
    "ICH_Q1B": "Q1B_Guideline.pdf",
    "ICH_Q1E": "Q1E_Guideline.pdf",
    "ICH_Q2": "ICH_Q2(R2)_Guideline_2023_1130.pdf",
    "ICH_Q6A": "Q6A_Guideline.pdf",
    "ICH_M3": "M3_R2__Guideline.pdf",
}
ALLOWED_GUIDELINE_IDS = set(GUIDELINE_SOURCE_FILES)
ALLOWED_GUIDELINE_FILES = set(GUIDELINE_SOURCE_FILES.values())

GUIDELINE_REGISTRY = {
    "stability": ("ICH_Q1A", ["2.1.7", "2.1.8", "2.2.1", "2.2.4", "3.1.1"]),
    "extrapolation": ("ICH_Q1E", ["2.4", "2.5.1", "2.5.1.2", "2.6"]),
    "specifications": ("ICH_Q6A", ["3.1", "3.2", "4.1", "4.3"]),
    "quality_org": ("ICH_M4Q", ["3.2.S.2.1", "3.2.S.4.4", "3.2.P.3.1", "3.2.P.5.6", "3.2.P.8.2"]),
    "ctd_org": ("ICH_M4", ["II.3.2", "II.3.2.S", "II.3.2.P"]),
    "efficacy_summary": ("ICH_M4E", ["2.7.3.1", "2.7.3.2", "2.7.3.3"]),
    "csr_structure": ("ICH_E3", ["2", "9.5", "11.1", "12.2", "12.3.2", "16.1.7"]),
    "nonclinical": ("ICH_M4S", ["2.6", "2.6.2", "2.6.6"]),
}

DEMO_FILE_NAMES = {
    "label.pdf",
    "module_2_qos.pdf",
    "module_2_clinical_summary.pdf",
    "module_3_quality_stability.pdf",
    "module_3_manufacturing.pdf",
    "module_4_nonclinical_summary.pdf",
    "module_5_clinical_study_report.pdf",
}

PRODUCTS = [
    "Zolarin", "Trevax", "Cardiolen", "Prisantol", "Halbrutide",
    "Novoquet", "Marendazole", "Orphex", "Vendrel", "Quintapraz",
]

DOSAGE_FORMS = [
    "immediate-release tablet",
    "film-coated tablet",
    "oral capsule",
    "oral suspension",
    "modified-release tablet",
]

MANUFACTURERS = [
    "Apex Pharma Manufacturing Pvt. Ltd.",
    "Nova Labs Manufacturing Pvt. Ltd.",
    "Orion Pharma Manufacturing Ltd.",
    "Zenith Labs Pvt. Ltd.",
    "Helix Biopharma Site A",
    "Helix Biopharma Site B",
    "Solvex Pharmaceuticals GmbH",
    "Corvantis Labs S.A.",
]

SHELF_LIFE_VALUES = [12, 18, 24, 30, 36]
STORAGE_VALUES = ["Store below 25C", "Store below 30C", "Do not store above 25C", "Do not store above 30C", "Store at controlled room temperature"]
BATCHES = ["B001", "B002", "B003", "B004", "STB-101", "STB-102", "VAL-201", "VAL-202"]
ENDPOINTS = [
    "systolic blood pressure reduction",
    "diastolic blood pressure reduction",
    "heart rate reduction",
    "HbA1c reduction",
    "fasting glucose change",
    "patient-reported symptom score",
    "primary endpoint at Week 12",
    "primary endpoint at Day 28",
]
SAE_COUNTS = [0, 1, 2, 3, 4, 5, 6, 8]
CLAIMS = [
    ("rapid control of blood pressure within 24 hours", "Week 12"),
    ("symptom relief within 2 hours", "Week 8"),
    ("clinically meaningful improvement by Week 12", "Week 24"),
    ("reduced hospitalization risk", "exploratory safety follow-up"),
    ("improved tolerability profile", "general adverse event summary"),
]

DOSSIER_FILE_TEMPLATES = {
    "label": "label_variant_{n:03d}.pdf",
    "package_insert": "package_insert_variant_{n:03d}.pdf",
    "quality": "module_3_quality_variant_{n:03d}.pdf",
    "stability": "stability_report_variant_{n:03d}.pdf",
    "qos": "qos_variant_{n:03d}.pdf",
    "manufacturing": "manufacturing_variant_{n:03d}.pdf",
    "clinical_summary": "clinical_summary_variant_{n:03d}.pdf",
    "csr": "csr_variant_{n:03d}.pdf",
    "nonclinical": "nonclinical_summary_variant_{n:03d}.pdf",
}

REVIEWER_ACTIONS = {
    "missing_section": "Verify whether the required CTD section exists elsewhere; if absent, add the missing section with appropriate supporting content.",
    "shelf_life_mismatch": "Verify and align shelf-life information across stability documentation, QOS, and labeling.",
    "storage_condition_mismatch": "Verify and align storage-condition information across Module 3 stability documentation and labeling.",
    "manufacturer_mismatch": "Verify and align manufacturer names and sites across QOS and Module 3 manufacturing sections.",
    "batch_mismatch": "Verify batch identifiers across batch analysis, stability, and manufacturing sections and correct inconsistent references.",
    "endpoint_mismatch": "Verify the protocol-defined primary endpoint and align the clinical summary with the CSR.",
    "sae_count_mismatch": "Verify serious adverse event counts and reconcile safety reporting across the CSR and clinical safety summary.",
    "unsupported_claim": "Provide supporting clinical evidence for the claim or revise/remove the unsupported claim.",
    "weak_justification": "Review whether the justification is supported by adequate data, trend analysis, and documented rationale.",
    "no_gap": "No reviewer action is required based on the retrieved evidence.",
    "uncertain": "Review the original dossier source because retrieved evidence is ambiguous or partially conflicting.",
    "insufficient_evidence": "Retrieve the missing source section or ask the dossier owner for clarification before making a final determination.",
}


# =============================================================================
# 2. GENERIC HELPERS
# =============================================================================

_counter = 0


def next_id(prefix: str) -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter:06d}"


def rand_file(kind: str) -> str:
    n = random.randint(1, 999)
    return DOSSIER_FILE_TEMPLATES[kind].format(n=n)


def rand_score(low: float = 0.68, high: float = 0.96) -> float:
    return round(random.uniform(low, high), 4)


def page() -> int:
    return random.randint(1, 80)


def source_hash(file_name: str) -> str:
    safe = file_name.replace(".", "_").replace("-", "_")
    return f"synthetic_hash_{safe}_{random.randint(1000, 9999)}"


def chunk_id(prefix: str) -> str:
    safe = prefix.replace(".", "_").replace("-", "_").replace("/", "_")
    return f"{safe}_chunk_{random.randint(1, 250):04d}"


def guideline_source_file(guideline_id: str) -> str:
    if guideline_id not in GUIDELINE_SOURCE_FILES:
        raise ValueError(f"Unknown guideline_id: {guideline_id}")
    return GUIDELINE_SOURCE_FILES[guideline_id]


def guideline_evidence(guideline_id: str, section: str, text: str, score: float | None = None) -> dict[str, Any]:
    return {
        "source_type": "guideline",
        "guideline_id": guideline_id,
        "source_file": guideline_source_file(guideline_id),
        "page_number": page(),
        "section": section,
        "chunk_id": chunk_id(guideline_id),
        "text": text,
        "retrieval_score": rand_score() if score is None else round(score, 4),
        "evidence_status": "strong" if (score or 0.9) >= 0.75 else "medium",
    }


def dossier_evidence(file_name: str, document_type: str, module_guess: str, section: str | None, text: str, score: float | None = None) -> dict[str, Any]:
    return {
        "source_type": "dossier",
        "file_name": file_name,
        "document_type": document_type,
        "module_guess": module_guess,
        "page_number": page(),
        "section": section,
        "chunk_id": chunk_id(file_name),
        "source_hash": source_hash(file_name),
        "text": text,
        "retrieval_score": rand_score() if score is None else round(score, 4),
        "evidence_status": "strong" if (score or 0.9) >= 0.75 else "medium",
    }


def guideline_citation(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "guideline_id": ev["guideline_id"],
        "source_file": ev["source_file"],
        "page_number": ev["page_number"],
        "section": ev["section"],
        "chunk_id": ev["chunk_id"],
    }


def dossier_citation(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": ev["file_name"],
        "page_number": ev["page_number"],
        "section": ev["section"],
        "chunk_id": ev["chunk_id"],
    }


def make_decision(
    llm_decision: str,
    finding_type: str,
    severity: str,
    evidence_status: str,
    ctd_section: str | None,
    finding_summary: str,
    reasoning_summary: str,
    guideline_citations: list[dict[str, Any]],
    dossier_citations: list[dict[str, Any]],
    reviewer_action: str,
    needs_rule_verification: bool,
) -> dict[str, Any]:
    return {
        "llm_decision": llm_decision,
        "finding_type": finding_type,
        "severity": severity,
        "evidence_status": evidence_status,
        "ctd_section": ctd_section,
        "finding_summary": finding_summary,
        "reasoning_summary": reasoning_summary,
        "guideline_citations": guideline_citations,
        "dossier_citations": dossier_citations,
        "reviewer_action": reviewer_action,
        "needs_rule_verification": needs_rule_verification,
    }


def d2g_package(
    package_id: str,
    fact_type: str,
    fact_text: str,
    extracted_value: str,
    fact_source: dict[str, Any],
    guideline_items: list[dict[str, Any]],
    cross_items: list[dict[str, Any]],
    guideline_strength: str = "strong",
    cross_strength: str = "strong",
) -> dict[str, Any]:
    return {
        "package_id": package_id,
        "direction": "dossier_to_guideline",
        "source_type": "dossier_fact",
        "dossier_fact": {
            "fact_id": f"FACT-{package_id}",
            "fact_type": fact_type,
            "extracted_value": extracted_value,
            "fact_text": fact_text,
            "source": {
                "file_name": fact_source["file_name"],
                "document_type": fact_source["document_type"],
                "module_guess": fact_source["module_guess"],
                "page_number": fact_source["page_number"],
                "section": fact_source["section"],
                "chunk_id": fact_source["chunk_id"],
                "source_hash": fact_source["source_hash"],
            },
        },
        "guideline_evidence_strength": guideline_strength,
        "guideline_evidence": guideline_items,
        "cross_dossier_evidence_strength": cross_strength,
        "cross_dossier_evidence": cross_items,
        "task": "Assess whether the dossier fact, claim, value, endpoint, count, or entity is supported by relevant guideline expectations and consistent with other retrieved dossier evidence.",
    }


def g2d_package(
    package_id: str,
    guideline_requirement: dict[str, Any],
    dossier_items: list[dict[str, Any]],
    dossier_strength: str,
) -> dict[str, Any]:
    return {
        "package_id": package_id,
        "direction": "guideline_to_dossier",
        "source_type": "guideline_requirement",
        "guideline_requirement": {
            "guideline_id": guideline_requirement["guideline_id"],
            "guideline_version": "synthetic_v3",
            "section": guideline_requirement["section"],
            "title": "Synthetic CTD requirement",
            "domain": "quality" if guideline_requirement["guideline_id"] in {"ICH_M4Q", "ICH_Q1A", "ICH_Q1E", "ICH_Q6A"} else "clinical",
            "source_file": guideline_requirement["source_file"],
            "page_number": guideline_requirement["page_number"],
            "chunk_id": guideline_requirement["chunk_id"],
            "requirement_text": guideline_requirement["text"],
        },
        "dossier_evidence_strength": dossier_strength,
        "dossier_evidence": dossier_items,
        "task": "Assess whether the retrieved dossier evidence adequately addresses the guideline requirement.",
    }


def make_chat_example(package: dict[str, Any], decision: dict[str, Any], issue_family: str, difficulty: str) -> dict[str, Any]:
    example_id = next_id(issue_family)
    user_content = (
        "Review the following evidence package and return only valid JSON using the required schema.\n\n"
        "Evidence package:\n"
        + json.dumps(package, ensure_ascii=False, indent=2)
    )
    assistant_content = json.dumps(decision, ensure_ascii=False)
    return {
        "example_id": example_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "dataset_version": DATASET_VERSION,
            "synthetic_label": issue_family,
            "issue_family": issue_family,
            "difficulty": difficulty,
            "expected_decision": decision["llm_decision"],
            "expected_finding_type": decision["finding_type"],
            "leakage_safe": True,
        },
    }


def random_guideline_topic(exclude: str | None = None) -> tuple[str, str, str]:
    topics = [t for t in GUIDELINE_REGISTRY if t != exclude]
    topic = random.choice(topics)
    guideline_id, sections = GUIDELINE_REGISTRY[topic]
    return topic, guideline_id, random.choice(sections)


def distractor_guideline(exclude: str | None = None) -> dict[str, Any]:
    topic, gid, section = random_guideline_topic(exclude)
    return guideline_evidence(
        gid,
        section,
        f"General guidance from {gid} section {section}; this text is related to dossier organization but does not resolve the main comparison.",
        score=random.uniform(0.30, 0.55),
    )


def distractor_dossier() -> dict[str, Any]:
    kind = random.choice(list(DOSSIER_FILE_TEMPLATES))
    file_name = rand_file(kind)
    return dossier_evidence(
        file_name=file_name,
        document_type="distractor",
        module_guess=random.choice(["Module 2", "Module 3", "Module 5"]),
        section=random.choice(["2.3", "3.2.P", "5.3.5.1", None]),
        text="Unrelated retrieved dossier excerpt that does not resolve the main regulatory comparison.",
        score=random.uniform(0.30, 0.55),
    )


# =============================================================================
# 3. TEMPLATE GENERATORS
# =============================================================================


def generate_missing_section(n: int) -> list[dict[str, Any]]:
    examples = []
    candidates = [
        ("ICH_M4Q", "3.2.P.8.2", "Post-approval Stability Protocol and Stability Commitment", "3.2.P.8.2"),
        ("ICH_E3", "2", "CSR Synopsis", "5.3.5.1"),
        ("ICH_M4Q", "3.2.P.5.6", "Justification of Specification", "3.2.P.5.6"),
        ("ICH_M4E", "2.7.3", "Summary of Clinical Efficacy", "2.7.3"),
        ("ICH_M4S", "2.6.6", "Toxicology Written Summary", "2.6.6"),
    ]
    for i in range(n):
        gid, section, title, ctd_section = candidates[i % len(candidates)]
        product = random.choice(PRODUCTS)
        g = guideline_evidence(gid, section, f"The submission should include {title} for {product} in CTD section {ctd_section}.")
        dossier_items = [distractor_dossier() for _ in range(random.randint(1, 3))]
        package = g2d_package(f"PKG-V3-{_counter + 1:05d}", g, dossier_items, "missing")
        decision = make_decision(
            "gap",
            "missing_section",
            "high",
            "missing",
            ctd_section,
            f"Required CTD section {ctd_section} is not supported by retrieved dossier evidence.",
            f"Guideline evidence expects {title}, but retrieval found only unrelated dossier content and no direct {ctd_section} section.",
            [guideline_citation(g)],
            [],
            REVIEWER_ACTIONS["missing_section"],
            True,
        )
        examples.append(make_chat_example(package, decision, "missing_section", "medium"))
    return examples


def generate_shelf_life_mismatch(n: int) -> list[dict[str, Any]]:
    examples = []
    for i in range(n):
        product = random.choice(PRODUCTS)
        form = random.choice(DOSAGE_FORMS)
        if i == 0:
            label_months, module_months = 18, 24
        else:
            label_months, module_months = random.sample(SHELF_LIFE_VALUES, 2)
        label_file = rand_file("label")
        module_file = rand_file("stability")
        label = dossier_evidence(label_file, "labeling", "Module 1", None, f"Shelf Life: {label_months} months for {product} {form}.")
        module = dossier_evidence(module_file, "quality_stability", "Module 3", "3.2.P.8.1", f"The proposed shelf life for {product} {form} is {module_months} months based on stability data.")
        g = guideline_evidence("ICH_Q1E", "2.5.1", "Shelf-life proposals should be supported by stability data and consistently reflected across submission documents.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "shelf_life_statement", label["text"], f"{label_months} months", label, [g, distractor_guideline("extrapolation")], [module], "strong", "strong")
        decision = make_decision(
            "gap", "value_inconsistency", "high", "strong", "3.2.P.8.1",
            "Shelf life differs between labeling and Module 3 stability evidence.",
            f"Labeling states {label_months} months, while Module 3 stability evidence states {module_months} months for the same product.",
            [guideline_citation(g)], [dossier_citation(label), dossier_citation(module)], REVIEWER_ACTIONS["shelf_life_mismatch"], True,
        )
        examples.append(make_chat_example(package, decision, "shelf_life_mismatch", "easy" if i == 0 else "medium"))
    return examples


def generate_storage_condition_mismatch(n: int) -> list[dict[str, Any]]:
    examples = []
    for i in range(n):
        product = random.choice(PRODUCTS)
        if i == 0:
            label_storage, module_storage = "Store below 30C", "Store below 25C"
        else:
            label_storage, module_storage = random.sample(STORAGE_VALUES, 2)
        label = dossier_evidence(rand_file("label"), "labeling", "Module 1", None, f"Storage condition: {label_storage}.")
        module = dossier_evidence(rand_file("quality"), "quality_stability", "Module 3", "3.2.P.8.1", f"Stability conclusions for {product}: {module_storage}.")
        g = guideline_evidence("ICH_Q1A", "2.2.4", "Storage conditions used in labeling should be supported by stability data and consistently stated.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "storage_condition_statement", label["text"], label_storage, label, [g], [module, distractor_dossier()], "strong", "strong")
        decision = make_decision(
            "gap", "value_inconsistency", "high", "strong", "3.2.P.8.1",
            "Storage condition differs between labeling and Module 3 stability evidence.",
            f"Labeling states '{label_storage}', while Module 3 stability evidence states '{module_storage}'. These storage statements are not equivalent.",
            [guideline_citation(g)], [dossier_citation(label), dossier_citation(module)], REVIEWER_ACTIONS["storage_condition_mismatch"], True,
        )
        examples.append(make_chat_example(package, decision, "storage_condition_mismatch", "easy" if i == 0 else "medium"))
    return examples


def generate_manufacturer_mismatch(n: int) -> list[dict[str, Any]]:
    examples = []
    for i in range(n):
        product = random.choice(PRODUCTS)
        if i % 10 == 0:
            m1, m2, severity = "Nova Labs Pvt. Ltd.", "Nova Labs Private Limited", "low"
        else:
            m1, m2 = random.sample(MANUFACTURERS, 2)
            severity = "medium"
        qos = dossier_evidence(rand_file("qos"), "quality_overall_summary", "Module 2", "2.3", f"QOS lists the manufacturing site for {product} as {m1}.")
        mfg = dossier_evidence(rand_file("manufacturing"), "manufacturing", "Module 3", "3.2.P.3.1", f"Drug product manufacturer: {m2}.")
        g = guideline_evidence("ICH_M4Q", "3.2.P.3.1", "Manufacturer names and sites should be identified consistently in quality documentation.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "manufacturer_statement", qos["text"], m1, qos, [g], [mfg], "strong", "strong")
        decision = make_decision(
            "gap", "entity_inconsistency", severity, "strong", "3.2.P.3.1",
            "Manufacturer identity differs across quality documents.",
            f"QOS identifies '{m1}', while Module 3 manufacturing evidence identifies '{m2}'. The names are not consistently presented.",
            [guideline_citation(g)], [dossier_citation(qos), dossier_citation(mfg)], REVIEWER_ACTIONS["manufacturer_mismatch"], True,
        )
        examples.append(make_chat_example(package, decision, "manufacturer_mismatch", "hard" if severity == "low" else "medium"))
    return examples


def generate_batch_mismatch(n: int) -> list[dict[str, Any]]:
    examples = []
    for _ in range(n):
        product = random.choice(PRODUCTS)
        primary = random.choice(BATCHES)
        alternatives = random.sample([b for b in BATCHES if b != primary], 2)
        qos = dossier_evidence(rand_file("qos"), "quality_overall_summary", "Module 2", "2.3", f"Primary commercial stability batch for {product}: {primary}.")
        mfg = dossier_evidence(rand_file("manufacturing"), "manufacturing", "Module 3", "3.2.P.3.5", f"Process validation batches listed: {alternatives[0]} and {alternatives[1]}.")
        g = guideline_evidence("ICH_M4Q", "3.2.P.3.5", "Batch references across quality sections should be traceable and reconciled.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "batch_identifier", qos["text"], primary, qos, [g], [mfg], "strong", "strong")
        decision = make_decision(
            "gap", "value_inconsistency", "medium", "strong", "3.2.P.3.5",
            "Batch identifiers are inconsistent across quality sections.",
            f"QOS references batch {primary}, while the manufacturing section lists {alternatives[0]} and {alternatives[1]} without reconciling {primary}.",
            [guideline_citation(g)], [dossier_citation(qos), dossier_citation(mfg)], REVIEWER_ACTIONS["batch_mismatch"], True,
        )
        examples.append(make_chat_example(package, decision, "batch_mismatch", "medium"))
    return examples


def generate_endpoint_mismatch(n: int) -> list[dict[str, Any]]:
    examples = []
    for _ in range(n):
        e1, e2 = random.sample(ENDPOINTS, 2)
        protocol = dossier_evidence(rand_file("csr"), "study_protocol", "Module 5", "5.3.5.1", f"Protocol-defined primary endpoint: {e1}.")
        summary = dossier_evidence(rand_file("clinical_summary"), "clinical_summary", "Module 2", "2.7.3", f"Clinical summary states the primary endpoint was {e2}.")
        g = guideline_evidence("ICH_E3", "9.5", "The CSR and clinical summary should consistently describe the protocol-defined endpoints.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "primary_endpoint", summary["text"], e2, summary, [g], [protocol], "strong", "strong")
        decision = make_decision(
            "gap", "endpoint_inconsistency", "high", "strong", "2.7.3",
            "Primary endpoint differs between protocol/CSR evidence and clinical summary.",
            f"The protocol identifies '{e1}', while the clinical summary states '{e2}', creating a direct endpoint-name mismatch.",
            [guideline_citation(g)], [dossier_citation(protocol), dossier_citation(summary)], REVIEWER_ACTIONS["endpoint_mismatch"], True,
        )
        examples.append(make_chat_example(package, decision, "endpoint_mismatch", "medium"))
    return examples


def generate_sae_count_mismatch(n: int) -> list[dict[str, Any]]:
    examples = []
    for _ in range(n):
        c1, c2 = random.sample(SAE_COUNTS, 2)
        csr = dossier_evidence(rand_file("csr"), "clinical_study_report", "Module 5", "12.3.2", f"The CSR reports {c1} serious adverse events.")
        safety = dossier_evidence(rand_file("clinical_summary"), "clinical_safety_summary", "Module 2", "2.7.4", f"The Summary of Clinical Safety reports {c2} serious adverse events.")
        g = guideline_evidence("ICH_E3", "12.3.2", "Safety results, including serious adverse event counts, should reconcile across clinical documents.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "sae_count", safety["text"], str(c2), safety, [g], [csr], "strong", "strong")
        decision = make_decision(
            "gap", "safety_count_inconsistency", "high", "strong", "2.7.4",
            "Serious adverse event counts differ across clinical safety documents.",
            f"The CSR reports {c1} SAEs, while the clinical safety summary reports {c2} SAEs.",
            [guideline_citation(g)], [dossier_citation(csr), dossier_citation(safety)], REVIEWER_ACTIONS["sae_count_mismatch"], True,
        )
        examples.append(make_chat_example(package, decision, "sae_count_mismatch", "medium"))
    return examples


def generate_unsupported_claim(n: int) -> list[dict[str, Any]]:
    examples = []
    for i in range(n):
        product = random.choice(PRODUCTS)
        if i == 0:
            claim, support = "rapid control of blood pressure within 24 hours", "Week 12"
        else:
            claim, support = random.choice(CLAIMS)
        label = dossier_evidence(rand_file("label"), "labeling", "Module 1", None, f"Label claim: {product} provides {claim}.")
        csr = dossier_evidence(rand_file("csr"), "clinical_study_report", "Module 5", "5.3.5.1", f"CSR reports efficacy assessment at {support}; no evidence directly supports the specific claim timeframe.")
        g = guideline_evidence("ICH_M4E", "2.7.3", "Clinical claims in labeling should be supported by relevant efficacy evidence in the clinical summary and CSR.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "label_claim_statement", label["text"], claim, label, [g], [csr], "strong", "medium")
        decision = make_decision(
            "gap", "unsupported_claim", "medium", "medium", "2.7.3",
            "Label claim is not clearly supported by retrieved clinical evidence.",
            f"The label claims {claim}, but the retrieved CSR evidence only discusses {support} and does not support the specific claim language.",
            [guideline_citation(g)], [dossier_citation(label), dossier_citation(csr)], REVIEWER_ACTIONS["unsupported_claim"], True,
        )
        examples.append(make_chat_example(package, decision, "unsupported_claim", "easy" if i == 0 else "medium"))
    return examples


def generate_weak_justification(n: int) -> list[dict[str, Any]]:
    examples = []
    weak_phrases = [
        "justified based on limited available data",
        "considered acceptable based on manufacturing experience",
        "supported by internal historical knowledge",
        "briefly justified without trend discussion",
    ]
    for _ in range(n):
        product = random.choice(PRODUCTS)
        phrase = random.choice(weak_phrases)
        dossier = dossier_evidence(rand_file("quality"), "quality_stability", "Module 3", "3.2.P.8.1", f"The proposed shelf life for {product} is {phrase}.")
        g = guideline_evidence("ICH_Q1E", "2.5.1", "Shelf-life justification should include adequate stability data, trend evaluation, and scientific rationale.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "shelf_life_justification", dossier["text"], phrase, dossier, [g], [dossier], "strong", "medium")
        decision = make_decision(
            "gap", "weak_justification", "medium", "medium", "3.2.P.8.1",
            "Shelf-life justification appears weak based on retrieved evidence.",
            f"The dossier gives only a brief justification ({phrase}) while guideline evidence expects data-supported trend evaluation.",
            [guideline_citation(g)], [dossier_citation(dossier)], REVIEWER_ACTIONS["weak_justification"], False,
        )
        examples.append(make_chat_example(package, decision, "weak_justification", "medium"))
    return examples


def generate_no_gap_quality(n: int) -> list[dict[str, Any]]:
    examples = []
    scenarios = ["shelf", "storage", "manufacturer", "batch", "section_present"]
    for _ in range(n):
        product = random.choice(PRODUCTS)
        scenario = random.choice(scenarios)
        if scenario == "shelf":
            months = random.choice(SHELF_LIFE_VALUES)
            label = dossier_evidence(rand_file("label"), "labeling", "Module 1", None, f"Shelf Life: {months} months.")
            module = dossier_evidence(rand_file("stability"), "quality_stability", "Module 3", "3.2.P.8.1", f"Proposed shelf life is {months} months for {product}.")
            g = guideline_evidence("ICH_Q1E", "2.5.1", "Shelf-life statements should be supported and consistent.")
            ctd = "3.2.P.8.1"
            summary = "Shelf-life statements are consistent across retrieved evidence."
            reason = f"Both labeling and Module 3 state {months} months."
            citations = [dossier_citation(label), dossier_citation(module)]
            package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "shelf_life_statement", label["text"], f"{months} months", label, [g], [module], "strong", "strong")
        elif scenario == "storage":
            storage = random.choice(STORAGE_VALUES)
            label = dossier_evidence(rand_file("label"), "labeling", "Module 1", None, f"Storage: {storage}.")
            module = dossier_evidence(rand_file("quality"), "quality_stability", "Module 3", "3.2.P.8.1", f"Stability conclusion: {storage}.")
            g = guideline_evidence("ICH_Q1A", "2.2.4", "Storage condition should be supported by stability evidence.")
            ctd = "3.2.P.8.1"
            summary = "Storage condition is consistent across retrieved evidence."
            reason = f"Both labeling and Module 3 state '{storage}'."
            citations = [dossier_citation(label), dossier_citation(module)]
            package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "storage_condition_statement", label["text"], storage, label, [g], [module], "strong", "strong")
        elif scenario == "manufacturer":
            m = random.choice(MANUFACTURERS)
            qos = dossier_evidence(rand_file("qos"), "quality_overall_summary", "Module 2", "2.3", f"Manufacturer: {m}.")
            mfg = dossier_evidence(rand_file("manufacturing"), "manufacturing", "Module 3", "3.2.P.3.1", f"Drug product manufacturer: {m}.")
            g = guideline_evidence("ICH_M4Q", "3.2.P.3.1", "Manufacturer information should be consistently identified.")
            ctd = "3.2.P.3.1"
            summary = "Manufacturer information is consistent."
            reason = f"Both QOS and Module 3 identify {m}."
            citations = [dossier_citation(qos), dossier_citation(mfg)]
            package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "manufacturer_statement", qos["text"], m, qos, [g], [mfg], "strong", "strong")
        elif scenario == "batch":
            b = random.choice(BATCHES)
            qos = dossier_evidence(rand_file("qos"), "quality_overall_summary", "Module 2", "2.3", f"Primary stability batch: {b}.")
            analysis = dossier_evidence(rand_file("quality"), "batch_analysis", "Module 3", "3.2.P.5.4", f"Batch analysis includes batch {b}.")
            g = guideline_evidence("ICH_M4Q", "3.2.P.5.4", "Batch references should be traceable and consistent.")
            ctd = "3.2.P.5.4"
            summary = "Batch identifiers are consistent."
            reason = f"Both retrieved sections reference batch {b}."
            citations = [dossier_citation(qos), dossier_citation(analysis)]
            package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "batch_identifier", qos["text"], b, qos, [g], [analysis], "strong", "strong")
        else:
            g = guideline_evidence("ICH_M4Q", "3.2.P.8.2", "The dossier should include a post-approval stability protocol and commitment.")
            present = dossier_evidence(rand_file("stability"), "quality_stability", "Module 3", "3.2.P.8.2", "3.2.P.8.2 Post-approval Stability Protocol and Stability Commitment is provided with annual commitment batches.")
            ctd = "3.2.P.8.2"
            summary = "Required stability protocol section is present."
            reason = "The retrieved dossier evidence directly includes section 3.2.P.8.2."
            citations = [dossier_citation(present)]
            package = g2d_package(f"PKG-V3-{_counter + 1:05d}", g, [present], "strong")
        decision = make_decision(
            "no_gap", "no_gap", "none", "strong", ctd, summary, reason,
            [guideline_citation(g)], citations, REVIEWER_ACTIONS["no_gap"], False,
        )
        examples.append(make_chat_example(package, decision, "no_gap_quality", "hard"))
    return examples


def generate_no_gap_clinical(n: int) -> list[dict[str, Any]]:
    examples = []
    scenarios = ["endpoint", "sae", "claim", "csr_synopsis"]
    for _ in range(n):
        product = random.choice(PRODUCTS)
        scenario = random.choice(scenarios)
        if scenario == "endpoint":
            endpoint = random.choice(ENDPOINTS)
            protocol = dossier_evidence(rand_file("csr"), "study_protocol", "Module 5", "5.3.5.1", f"Protocol-defined primary endpoint: {endpoint}.")
            summary = dossier_evidence(rand_file("clinical_summary"), "clinical_summary", "Module 2", "2.7.3", f"The primary endpoint was {endpoint}.")
            g = guideline_evidence("ICH_E3", "9.5", "Primary endpoint reporting should be consistent.")
            ctd = "2.7.3"
            fs = "Primary endpoint is consistent across retrieved clinical evidence."
            rs = f"Both sources identify '{endpoint}' as the primary endpoint."
            dc = [dossier_citation(protocol), dossier_citation(summary)]
            package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "primary_endpoint", summary["text"], endpoint, summary, [g], [protocol], "strong", "strong")
        elif scenario == "sae":
            count = random.choice(SAE_COUNTS)
            csr = dossier_evidence(rand_file("csr"), "clinical_study_report", "Module 5", "12.3.2", f"The CSR reports {count} serious adverse events.")
            safety = dossier_evidence(rand_file("clinical_summary"), "clinical_safety_summary", "Module 2", "2.7.4", f"The clinical safety summary also reports {count} serious adverse events.")
            g = guideline_evidence("ICH_E3", "12.3.2", "SAE counts should reconcile across clinical documents.")
            ctd = "2.7.4"
            fs = "Serious adverse event counts are consistent."
            rs = f"Both CSR and clinical safety summary report {count} SAEs."
            dc = [dossier_citation(csr), dossier_citation(safety)]
            package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "sae_count", safety["text"], str(count), safety, [g], [csr], "strong", "strong")
        elif scenario == "claim":
            claim, _support = random.choice(CLAIMS)
            label = dossier_evidence(rand_file("label"), "labeling", "Module 1", None, f"Label claim: {product} provides {claim}.")
            csr = dossier_evidence(rand_file("csr"), "clinical_study_report", "Module 5", "5.3.5.1", f"CSR includes a prespecified analysis directly supporting the claim that {product} provides {claim}.")
            g = guideline_evidence("ICH_M4E", "2.7.3", "Clinical claims should be supported by relevant clinical evidence.")
            ctd = "2.7.3"
            fs = "Label claim is supported by retrieved clinical evidence."
            rs = "The retrieved CSR evidence directly supports the label claim."
            dc = [dossier_citation(label), dossier_citation(csr)]
            package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "label_claim_statement", label["text"], claim, label, [g], [csr], "strong", "strong")
        else:
            g = guideline_evidence("ICH_E3", "2", "The clinical study report should include a synopsis.")
            synopsis = dossier_evidence(rand_file("csr"), "clinical_study_report", "Module 5", "Synopsis", "CSR Synopsis section is present and summarizes study design, efficacy, and safety results.")
            ctd = "5.3.5.1"
            fs = "CSR synopsis is present in retrieved evidence."
            rs = "The retrieved CSR evidence contains a dedicated Synopsis section."
            dc = [dossier_citation(synopsis)]
            package = g2d_package(f"PKG-V3-{_counter + 1:05d}", g, [synopsis], "strong")
        decision = make_decision(
            "no_gap", "no_gap", "none", "strong", ctd, fs, rs,
            [guideline_citation(g)], dc, REVIEWER_ACTIONS["no_gap"], False,
        )
        examples.append(make_chat_example(package, decision, "no_gap_clinical", "hard"))
    return examples


def generate_uncertain_ambiguous_evidence(n: int) -> list[dict[str, Any]]:
    examples = []
    for _ in range(n):
        product = random.choice(PRODUCTS)
        fact = dossier_evidence(rand_file("quality"), "quality_stability", "Module 3", "3.2.P.8.1", f"Stability section for {product} states storage should follow controlled conditions, but the exact limit is not clearly shown in the retrieved text.", score=random.uniform(0.50, 0.70))
        cross = dossier_evidence(rand_file("label"), "labeling", "Module 1", None, "Labeling excerpt mentions storage but the numeric condition is partly truncated.", score=random.uniform(0.45, 0.65))
        g = guideline_evidence("ICH_Q1A", "2.2.4", "Storage conditions should be clearly supported by stability data.")
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "storage_condition_statement", fact["text"], "ambiguous storage condition", fact, [g], [cross], "medium", "weak")
        decision = make_decision(
            "uncertain", "other", "low", "weak", "3.2.P.8.1",
            "Retrieved storage evidence is ambiguous and not enough to confirm a gap.",
            "The retrieved evidence suggests a possible storage-condition issue, but the exact numeric storage limit is unclear or truncated, so the model should not claim a confirmed mismatch.",
            [guideline_citation(g)], [dossier_citation(fact), dossier_citation(cross)], REVIEWER_ACTIONS["uncertain"], True,
        )
        examples.append(make_chat_example(package, decision, "uncertain_ambiguous_evidence", "hard"))
    return examples


def generate_insufficient_evidence(n: int) -> list[dict[str, Any]]:
    examples = []
    reasons = [
        "only a section heading was retrieved",
        "the key value is cut off mid-sentence",
        "OCR text is unreadable at the relevant line",
        "the linked page is missing from the indexed dossier corpus",
    ]
    for _ in range(n):
        product = random.choice(PRODUCTS)
        reason = random.choice(reasons)
        g = guideline_evidence("ICH_Q1E", "2.5.1", "Shelf-life and storage claims should be supported by adequate stability evidence.")
        weak = dossier_evidence(rand_file("stability"), "quality_stability", "Module 3", "3.2.P.8.1", f"Stability summary for {product}: [incomplete text; {reason}].", score=random.uniform(0.35, 0.55))
        package = d2g_package(f"PKG-V3-{_counter + 1:05d}", "stability_statement", weak["text"], "not available", weak, [g], [], "medium", "missing")
        decision = make_decision(
            "needs_human_review", "insufficient_evidence", "medium", "missing", "3.2.P.8.1",
            "Retrieved evidence is insufficient for a final gap/no-gap decision.",
            f"The only retrieved dossier evidence is incomplete because {reason}; no reliable source chunk provides the missing value.",
            [guideline_citation(g)], [dossier_citation(weak)], REVIEWER_ACTIONS["insufficient_evidence"], False,
        )
        examples.append(make_chat_example(package, decision, "insufficient_evidence", "hard"))
    return examples


GENERATORS = {
    "missing_section": generate_missing_section,
    "shelf_life_mismatch": generate_shelf_life_mismatch,
    "storage_condition_mismatch": generate_storage_condition_mismatch,
    "manufacturer_mismatch": generate_manufacturer_mismatch,
    "batch_mismatch": generate_batch_mismatch,
    "endpoint_mismatch": generate_endpoint_mismatch,
    "sae_count_mismatch": generate_sae_count_mismatch,
    "unsupported_claim": generate_unsupported_claim,
    "weak_justification": generate_weak_justification,
    "no_gap_quality": generate_no_gap_quality,
    "no_gap_clinical": generate_no_gap_clinical,
    "uncertain_ambiguous_evidence": generate_uncertain_ambiguous_evidence,
    "insufficient_evidence": generate_insufficient_evidence,
}


# =============================================================================
# 4. VALIDATION AND WRITING
# =============================================================================


def assistant_json(example: dict[str, Any]) -> dict[str, Any]:
    return json.loads(example["messages"][2]["content"])


def validate_decision(decision: dict[str, Any], example_id: str) -> list[str]:
    errors = []
    keys = set(decision)
    expected = set(ASSISTANT_KEYS)
    if keys != expected:
        errors.append(f"{example_id}: assistant keys mismatch. got={sorted(keys)} expected={sorted(expected)}")

    if decision.get("llm_decision") not in ALLOWED_DECISIONS:
        errors.append(f"{example_id}: invalid llm_decision {decision.get('llm_decision')!r}")
    if decision.get("finding_type") not in ALLOWED_FINDING_TYPES:
        errors.append(f"{example_id}: invalid finding_type {decision.get('finding_type')!r}")
    if decision.get("severity") not in ALLOWED_SEVERITIES:
        errors.append(f"{example_id}: invalid severity {decision.get('severity')!r}")
    if decision.get("evidence_status") not in ALLOWED_EVIDENCE_STATUS:
        errors.append(f"{example_id}: invalid evidence_status {decision.get('evidence_status')!r}")
    if not isinstance(decision.get("needs_rule_verification"), bool):
        errors.append(f"{example_id}: needs_rule_verification must be boolean")

    if not str(decision.get("reviewer_action", "")).strip():
        errors.append(f"{example_id}: reviewer_action is blank")
    if not str(decision.get("finding_summary", "")).strip():
        errors.append(f"{example_id}: finding_summary is blank")
    if not str(decision.get("reasoning_summary", "")).strip():
        errors.append(f"{example_id}: reasoning_summary is blank")

    if decision.get("llm_decision") == "no_gap":
        if decision.get("finding_type") != "no_gap":
            errors.append(f"{example_id}: no_gap decision must use finding_type no_gap")
        if decision.get("severity") != "none":
            errors.append(f"{example_id}: no_gap decision must use severity none")
    if decision.get("finding_type") in GAP_FINDING_TYPES and decision.get("llm_decision") == "no_gap":
        errors.append(f"{example_id}: gap finding_type cannot use llm_decision no_gap")
    if decision.get("finding_type") in GAP_FINDING_TYPES and decision.get("llm_decision") != "gap":
        errors.append(f"{example_id}: gap finding_type should use llm_decision gap")

    for i, cite in enumerate(decision.get("guideline_citations", [])):
        if cite.get("guideline_id") not in ALLOWED_GUIDELINE_IDS:
            errors.append(f"{example_id}: guideline citation {i} has invalid guideline_id {cite.get('guideline_id')!r}")
        if cite.get("source_file") not in ALLOWED_GUIDELINE_FILES:
            errors.append(f"{example_id}: guideline citation {i} has invalid source_file {cite.get('source_file')!r}")
        if cite.get("source_file") in DEMO_FILE_NAMES:
            errors.append(f"{example_id}: guideline citation {i} contains demo dossier file")
        for field in ["page_number", "section", "chunk_id"]:
            if field not in cite:
                errors.append(f"{example_id}: guideline citation {i} missing {field}")

    for i, cite in enumerate(decision.get("dossier_citations", [])):
        file_name = cite.get("file_name")
        if file_name in ALLOWED_GUIDELINE_FILES:
            errors.append(f"{example_id}: dossier citation {i} cites guideline file")
        if file_name in DEMO_FILE_NAMES:
            errors.append(f"{example_id}: dossier citation {i} uses final demo file {file_name}")
        for field in ["file_name", "page_number", "chunk_id"]:
            if field not in cite:
                errors.append(f"{example_id}: dossier citation {i} missing {field}")

    return errors


def check_demo_leakage(example: dict[str, Any]) -> list[str]:
    text = json.dumps(example, ensure_ascii=False)
    hits = [name for name in DEMO_FILE_NAMES if name in text]
    return [f"{example['example_id']}: demo dossier leakage {name}" for name in hits]


def validate_examples(examples: list[dict[str, Any]]) -> None:
    errors = []
    seen_ids = set()
    for ex in examples:
        if ex["example_id"] in seen_ids:
            errors.append(f"duplicate example_id {ex['example_id']}")
        seen_ids.add(ex["example_id"])
        if [m.get("role") for m in ex.get("messages", [])] != ["system", "user", "assistant"]:
            errors.append(f"{ex['example_id']}: invalid chat roles")
        try:
            decision = assistant_json(ex)
        except json.JSONDecodeError as exc:
            errors.append(f"{ex['example_id']}: assistant content invalid JSON: {exc}")
            continue
        errors.extend(validate_decision(decision, ex["example_id"]))
        errors.extend(check_demo_leakage(ex))

    if errors:
        preview = "\n".join(errors[:40])
        raise ValueError(f"Validation failed with {len(errors)} errors. First errors:\n{preview}")


def distribution(examples: list[dict[str, Any]], field: str) -> dict[str, int]:
    if field == "issue_family":
        return dict(Counter(ex["metadata"]["issue_family"] for ex in examples))
    return dict(Counter(assistant_json(ex)[field] for ex in examples))


def print_distribution(examples: list[dict[str, Any]], name: str) -> None:
    print(f"\n{name}: {len(examples)} examples")
    for field in ["llm_decision", "finding_type", "severity", "evidence_status", "issue_family"]:
        print(f"\n{field} distribution")
        counts = distribution(examples, field)
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            pct = 100.0 * count / len(examples)
            print(f"  {key:35s} {count:4d} ({pct:5.1f}%)")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def manifest(train: list[dict[str, Any]], eval_: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    all_examples = train + eval_ + test
    return {
        "dataset_version": DATASET_VERSION,
        "total_examples": len(all_examples),
        "train_examples": len(train),
        "eval_examples": len(eval_),
        "test_examples": len(test),
        "random_seed": SEED,
        "created_for": "LLM regulatory dossier gap decision layer",
        "gold_issues_csv_used": False,
        "demo_dossier_leakage_detected": False,
        "decision_distribution": distribution(all_examples, "llm_decision"),
        "finding_type_distribution": distribution(all_examples, "finding_type"),
        "severity_distribution": distribution(all_examples, "severity"),
        "evidence_status_distribution": distribution(all_examples, "evidence_status"),
        "issue_family_distribution": distribution(all_examples, "issue_family"),
        "schema": {
            "llm_decision": sorted(ALLOWED_DECISIONS),
            "finding_type": sorted(ALLOWED_FINDING_TYPES),
            "severity": sorted(ALLOWED_SEVERITIES),
            "evidence_status": sorted(ALLOWED_EVIDENCE_STATUS),
        },
        "notes": [
            "Synthetic data only; final gold evaluation file was not used.",
            "Final demo dossier file names are blocked by validation.",
            "Guideline IDs use project registry names, including ICH_Q1A rather than ICH_Q1A_R2.",
        ],
    }


def error_targets() -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "targets": [
            {
                "bug_id": "v2_storage_mismatch_marked_uncertain",
                "fix": "Include repeated Store below 30C vs Store below 25C examples labeled gap/value_inconsistency/high.",
            },
            {
                "bug_id": "v2_unsupported_claim_confused_with_endpoint",
                "fix": "Include rapid-control label claim examples where CSR only has Week 12 evidence, labeled unsupported_claim rather than endpoint_inconsistency.",
            },
            {
                "bug_id": "v2_blank_reviewer_action",
                "fix": "Validation rejects every assistant JSON with empty reviewer_action.",
            },
            {
                "bug_id": "v2_invalid_guideline_citation_from_dossier",
                "fix": "Validation rejects guideline citations whose source_file is not a registered guideline file.",
            },
            {
                "bug_id": "v2_wrong_severity_for_shelf_life_or_storage",
                "fix": "Shelf-life and storage deterministic value mismatches are labeled high severity.",
            },
            {
                "bug_id": "v2_needs_hard_negative_no_gap_examples",
                "fix": "Dataset includes quality and clinical no-gap hard negatives with severity none and finding_type no_gap.",
            },
        ],
    }


def generate_all_examples() -> list[dict[str, Any]]:
    assert set(FAMILY_COUNTS) == ISSUE_FAMILIES
    assert sum(FAMILY_COUNTS.values()) == 1000
    all_examples: list[dict[str, Any]] = []
    for family, count in FAMILY_COUNTS.items():
        family_examples = GENERATORS[family](count)
        if len(family_examples) != count:
            raise RuntimeError(f"{family} generated {len(family_examples)} examples, expected {count}")
        all_examples.extend(family_examples)
    return all_examples


def main() -> None:
    global _counter
    _counter = 0
    random.seed(SEED)

    examples = generate_all_examples()
    validate_examples(examples)
    print(f"Schema validation passed for all {len(examples)} examples.")
    print("No demo dossier leakage detected.")

    random.Random(SEED).shuffle(examples)
    train = examples[:700]
    eval_ = examples[700:850]
    test = examples[850:1000]

    assert len(train) == 700
    assert len(eval_) == 150
    assert len(test) == 150

    write_jsonl(train, TRAIN_PATH)
    write_jsonl(eval_, EVAL_PATH)
    write_jsonl(test, TEST_PATH)
    write_json(manifest(train, eval_, test), MANIFEST_PATH)
    write_json(error_targets(), ERROR_TARGETS_PATH)

    print_distribution(examples, "Full dataset")
    print("\nOutput files")
    print(f"  Train:    {TRAIN_PATH}")
    print(f"  Eval:     {EVAL_PATH}")
    print(f"  Test:     {TEST_PATH}")
    print(f"  Manifest: {MANIFEST_PATH}")
    print(f"  Targets:  {ERROR_TARGETS_PATH}")


if __name__ == "__main__":
    main()
