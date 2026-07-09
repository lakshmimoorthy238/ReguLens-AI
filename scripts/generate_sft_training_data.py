import json
import random
from itertools import product
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = PROJECT_ROOT / "training_data" / "v2"

TRAIN_PATH = V2_DIR / "llm_sft_train.jsonl"
EVAL_PATH = V2_DIR / "llm_sft_eval.jsonl"
TEST_PATH = V2_DIR / "llm_sft_test.jsonl"
MANIFEST_PATH = V2_DIR / "dataset_manifest.json"

RANDOM_SEED = 42
DATASET_VERSION = "v2"

ALLOWED_DECISIONS = {"gap", "no_gap", "uncertain", "needs_human_review"}
ALLOWED_SEVERITIES = {"high", "medium", "low", "none"}
ALLOWED_FINDING_TYPES = {
    "missing_section", "value_inconsistency", "entity_inconsistency",
    "endpoint_inconsistency", "safety_count_inconsistency", "unsupported_claim",
    "weak_justification", "insufficient_evidence", "no_gap", "other",
}
ALLOWED_EVIDENCE_STATUS = {"strong", "medium", "weak", "missing"}


SYSTEM_PROMPT = """
You are a regulatory dossier review assistant.

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
missing_section, value_inconsistency, entity_inconsistency, endpoint_inconsistency,
safety_count_inconsistency, unsupported_claim, weak_justification,
insufficient_evidence, no_gap, other
"""

# NOTE: deliberately does NOT include the actual demo product name
# ("Cardiostat") or any name/value used in sample_dossier / gold_issues.csv.
# Training data must stay disjoint from the held-out final demo/eval set —
# otherwise the model could partially memorize the exact demo answers
# instead of learning the general skill.
PRODUCTS = [
    "Neurostat", "Gastrozol", "Pulmocare", "Renacard", "Hepatrix",
    "Dermazine", "Osteofirm", "Glycomend", "Thyrolex", "Vasculin",
    "Nephrocare", "Bronchivex", "Immunolex", "Cortifen", "Analgex",
]

LABEL_FILE_POOL = [
    "label_variant_{n}.pdf", "package_insert_variant_{n}.pdf", "product_information_variant_{n}.pdf",
]
MODULE_FILE_POOL = [
    "module_3_quality_stability_variant_{n}.pdf", "quality_stability_report_variant_{n}.pdf",
    "module3_variant_{n}.pdf",
]

_file_counter = {"n": 0}


def next_file(pool: list[str]) -> str:
    _file_counter["n"] += 1
    return random.choice(pool).format(n=_file_counter["n"])


# ---------------------------------------------------------------------------
# Wording variation — same underlying value, different phrasing. Prevents
# the model from just pattern-matching one fixed sentence template per
# fact type rather than learning to read varied real-world phrasing.
# ---------------------------------------------------------------------------
def num_to_words(n: int) -> str:
    words = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        12: "twelve", 18: "eighteen", 20: "twenty", 24: "twenty-four",
        25: "twenty-five", 30: "thirty", 36: "thirty-six",
    }
    return words.get(n, str(n))


def shelf_life_phrase(months: int) -> str:
    templates = [
        f"Shelf Life: {months} months.",
        f"The proposed shelf life is {months} months.",
        f"The proposed shelf life is {num_to_words(months)} months.",
        f"Expiry period: {months} months.",
        f"The product has a shelf life of {months} months from the date of manufacture.",
    ]
    return random.choice(templates)


def storage_phrase(temp: int) -> str:
    templates = [
        f"Store below {temp}°C.",
        f"Storage Condition: Store below {temp}°C.",
        f"The recommended storage condition is store below {temp} degrees Celsius.",
        f"Storage: keep below {temp}°C, protected from light.",
    ]
    return random.choice(templates)


def sae_phrase(count: int) -> str:
    plural = "s" if count != 1 else ""
    templates = [
        f"{count} serious adverse event{plural}.",
        f"A total of {count} serious adverse event{plural} were reported.",
        f"{num_to_words(count).capitalize()} serious adverse event{plural} occurred during the treatment period.",
    ]
    return random.choice(templates)


def endpoint_intro(statement: str) -> str:
    templates = [
        f"The primary efficacy endpoint is {statement}.",
        f"Primary endpoint: {statement}.",
        f"The study's primary endpoint was defined as {statement}.",
        f"The primary efficacy measure reported is {statement}.",
    ]
    return random.choice(templates)


def endpoint_csr_intro(statement: str) -> str:
    templates = [
        f"The primary endpoint reported in the clinical study report is {statement}.",
        f"The CSR defines the primary endpoint as {statement}.",
        f"According to the CSR, the primary efficacy endpoint was {statement}.",
    ]
    return random.choice(templates)


def claim_intro(claim: str) -> str:
    templates = [
        claim,
        f"Labeling states: {claim}",
        f"According to the product label: {claim}",
        f"The label claims that {claim[0].lower()}{claim[1:]}",
    ]
    return random.choice(templates)


def justification_prefix(text: str) -> str:
    prefixes = ["", "Note: ", "Reviewer comment: ", "Observation: "]
    return f"{random.choice(prefixes)}{text}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def make_user_prompt(package: dict) -> str:
    return (
        "Review the following evidence package and return only valid JSON using the required schema.\n\n"
        "Evidence package:\n"
        f"{json.dumps(package, indent=2, ensure_ascii=False)}"
    )


def make_chat_example(package: dict, decision: dict, label: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": make_user_prompt(package)},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ],
        "metadata": {"synthetic_label": label},
    }


def guideline_citation(guideline_id, source_file, page_number, section, chunk_id):
    return {"guideline_id": guideline_id, "source_file": source_file, "page_number": page_number,
            "section": section, "chunk_id": chunk_id}


def dossier_citation(file_name, page_number, section, chunk_id):
    return {"file_name": file_name, "page_number": page_number, "section": section, "chunk_id": chunk_id}


def make_g2d_package(guideline_id, version, source_file, section, requirement_text, dossier_evidence):
    return {
        "package_id": "SYN-G2D",
        "direction": "guideline_to_dossier",
        "source_type": "guideline_requirement",
        "guideline_requirement": {
            "guideline_id": guideline_id, "guideline_version": version, "section": section,
            "title": "Synthetic guideline requirement", "domain": "Synthetic regulatory domain",
            "source_file": source_file, "page_number": random.randint(5, 60),
            "chunk_id": f"{guideline_id}_{section}_chunk", "requirement_text": requirement_text,
        },
        "dossier_evidence_strength": "medium",
        "dossier_evidence": dossier_evidence,
        "task": "Assess whether the retrieved dossier evidence adequately addresses the guideline requirement.",
    }


def make_d2g_package(fact_type, fact_text, source_file, guideline_evidence, cross_dossier_evidence):
    return {
        "package_id": "SYN-D2G",
        "direction": "dossier_to_guideline",
        "source_type": "dossier_fact",
        "dossier_fact": {
            "fact_id": "FACT-SYN", "fact_type": fact_type, "extracted_value": None, "fact_text": fact_text,
            "source": {"file_name": source_file, "document_type": "synthetic_document",
                       "module_guess": "Synthetic Module", "page_number": 1, "section": None,
                       "chunk_id": f"{source_file}_chunk_1", "source_hash": "synthetic_hash"},
        },
        "guideline_evidence_strength": "medium",
        "guideline_evidence": guideline_evidence,
        "cross_dossier_evidence_strength": "medium",
        "cross_dossier_evidence": cross_dossier_evidence,
        "task": ("Assess whether the dossier fact, claim, value, endpoint, count, or entity "
                 "is supported by relevant guideline expectations and consistent with other "
                 "retrieved dossier evidence."),
    }


def decision_json(llm_decision, finding_type, severity, evidence_status, ctd_section,
                   finding_summary, reasoning_summary, guideline_citations, dossier_citations, reviewer_action):
    return {
        "llm_decision": llm_decision, "finding_type": finding_type, "severity": severity,
        "evidence_status": evidence_status, "ctd_section": ctd_section,
        "finding_summary": finding_summary, "reasoning_summary": reasoning_summary,
        "guideline_citations": guideline_citations, "dossier_citations": dossier_citations,
        "reviewer_action": reviewer_action, "needs_rule_verification": True,
    }


def make_guideline_evidence(text, guideline_id, section, score=0.78):
    return [{"retrieval_score": score, "evidence_status": "strong" if score >= 0.75 else "medium",
             "guideline_id": guideline_id, "title": "Synthetic guideline evidence", "version": "R1",
             "domain": "Synthetic regulatory domain", "source_file": f"{guideline_id}_Guideline.pdf",
             "page_number": random.randint(5, 40), "section": section,
             "chunk_id": f"{guideline_id}_{section}_chunk", "text": text}]


def make_dossier_evidence(file_name, text, section=None, score=0.76):
    return {"retrieval_score": score,
            "evidence_status": "strong" if score >= 0.75 else ("medium" if score >= 0.5 else "weak"),
            "file_name": file_name, "document_type": "synthetic_document", "module_guess": "Synthetic Module",
            "page_number": 1, "section": section, "chunk_id": f"{file_name}_chunk_1",
            "source_hash": "synthetic_hash", "text": text}


def make_distractor_evidence():
    return make_dossier_evidence(
        "unrelated_appendix_variant.pdf",
        "This section discusses general administrative information and does not "
        "directly relate to the specific value or requirement in question.",
        section=None, score=0.35,
    )


def maybe_with_distractor(evidence_list, chance=0.4):
    if random.random() < chance:
        return evidence_list + [make_distractor_evidence()]
    return evidence_list


def rand_product():
    return random.choice(PRODUCTS)


# ---------------------------------------------------------------------------
# Category 1: manufacturer — match / formatting-only / true mismatch
# ---------------------------------------------------------------------------
MANUFACTURER_PAIRS = [
    ("Apex Pharma Manufacturing Pvt. Ltd.", "Apex Pharma Manufacturing Pvt. Ltd.", "match"),
    ("Nova Labs Manufacturing Pvt. Ltd.", "Nova Labs Manufacturing Pvt. Ltd.", "match"),
    ("Helix Biopharma Site A", "Helix Biopharma Site A", "match"),
    ("Summit Pharmaceuticals Ltd.", "Summit Pharmaceuticals Ltd.", "match"),
    ("Meridian Biologics Pvt. Ltd.", "Meridian Biologics Pvt. Ltd.", "match"),
    ("Coastal Pharma Works", "Coastal Pharma Works", "match"),
    ("Nova Labs Manufacturing Pvt. Ltd.", "Nova Labs Manufacturing Private Limited", "format"),
    ("Orion Pharma Mfg. Ltd.", "Orion Pharma Manufacturing Limited", "format"),
    ("Zenith Labs Pvt Ltd", "Zenith Labs Private Ltd.", "format"),
    ("Summit Pharmaceuticals Ltd.", "Summit Pharmaceutical Ltd.", "format"),
    ("Meridian Biologics Pvt. Ltd.", "Meridian Biologics Private Limited", "format"),
    ("Coastal Pharma Works", "Coastal Pharma Works Pvt. Ltd.", "format"),
    ("Orion Pharma Manufacturing Ltd.", "Zenith Labs Pvt. Ltd.", "mismatch"),
    ("Apex Pharma Manufacturing Pvt. Ltd.", "Nova Labs Manufacturing Pvt. Ltd.", "mismatch"),
    ("BlueRiver Biologics", "Crestline Pharmaceuticals", "mismatch"),
    ("Helix Biopharma Site A", "Helix Biopharma Site B", "mismatch"),
    ("Summit Pharmaceuticals Ltd.", "Meridian Biologics Pvt. Ltd.", "mismatch"),
    ("Coastal Pharma Works", "Highland Pharma Manufacturing", "mismatch"),
]


def generate_manufacturer_examples():
    examples = []
    for qos_name, module_name, kind in MANUFACTURER_PAIRS * 3:
        module_file = next_file(MODULE_FILE_POOL)
        package = make_d2g_package(
            fact_type="manufacturer_statement",
            fact_text=f"Manufacturer listed in the quality summary: {qos_name}.",
            source_file="module_2_qos_variant.pdf",
            guideline_evidence=make_guideline_evidence(
                "The dossier should identify drug product manufacturer and manufacturing site information consistently.",
                "ICH_M4Q", "3.2.P.3.1"),
            cross_dossier_evidence=maybe_with_distractor([
                make_dossier_evidence(module_file, f"The drug product manufacturer is: {module_name}.",
                                      section="3.2.P.3.1", score=0.82)]),
        )
        if kind == "match":
            decision = decision_json("no_gap", "no_gap", "none", "strong", "3.2.P.3.1",
                "Manufacturer information is consistent across retrieved evidence.",
                f"Both retrieved sources list {qos_name}.",
                [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 16, "3.2.P.3.1", "ICH_M4Q_3.2.P.3.1_chunk")],
                [dossier_citation("module_2_qos_variant.pdf", 1, None, "module_2_qos_variant.pdf_chunk_1"),
                 dossier_citation(module_file, 1, "3.2.P.3.1", f"{module_file}_chunk_1")],
                "No immediate action required for manufacturer consistency.")
        elif kind == "format":
            decision = decision_json("gap", "entity_inconsistency", "low", "strong", "3.2.P.3.1",
                "Manufacturer name differs only in formatting between sources.",
                f"Both sources refer to the same manufacturer, but wording differs ('{qos_name}' vs. '{module_name}') — a minor formatting inconsistency, not a substantive discrepancy.",
                [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 16, "3.2.P.3.1", "ICH_M4Q_3.2.P.3.1_chunk")],
                [dossier_citation("module_2_qos_variant.pdf", 1, None, "module_2_qos_variant.pdf_chunk_1"),
                 dossier_citation(module_file, 1, "3.2.P.3.1", f"{module_file}_chunk_1")],
                "Standardize manufacturer name formatting across documents for consistency.")
        else:
            decision = decision_json("gap", "entity_inconsistency", "high", "strong", "3.2.P.3.1",
                "Manufacturer information is inconsistent across quality documents.",
                f"The QOS lists {qos_name}, while Module 3 manufacturing evidence lists {module_name} — these are different entities, not a formatting variation.",
                [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 16, "3.2.P.3.1", "ICH_M4Q_3.2.P.3.1_chunk")],
                [dossier_citation("module_2_qos_variant.pdf", 1, None, "module_2_qos_variant.pdf_chunk_1"),
                 dossier_citation(module_file, 1, "3.2.P.3.1", f"{module_file}_chunk_1")],
                "Verify and align manufacturer information across QOS and Module 3.")
        examples.append(make_chat_example(package, decision, f"manufacturer_{kind}"))
    return examples


# ---------------------------------------------------------------------------
# Category 2: shelf life — grid x wording variants
# ---------------------------------------------------------------------------
MONTH_VALUES = [12, 18, 24, 30, 36]


def generate_shelf_life_examples():
    examples = []
    for label_months, module_months in product(MONTH_VALUES, MONTH_VALUES):
        for _ in range(3):  # 3 differently-worded examples per value pair
            is_gap = label_months != module_months
            diff = abs(label_months - module_months)
            severity = "none" if not is_gap else ("medium" if diff <= 6 else "high")
            label_file = next_file(LABEL_FILE_POOL)
            module_file = next_file(MODULE_FILE_POOL)
            label_text = shelf_life_phrase(label_months)
            module_text = shelf_life_phrase(module_months)

            package = make_d2g_package(
                fact_type="shelf_life_statement", fact_text=label_text, source_file=label_file,
                guideline_evidence=make_guideline_evidence(
                    "The stability summary should support the proposed shelf life and storage conditions.",
                    "ICH_Q1A_R2", "2.2"),
                cross_dossier_evidence=maybe_with_distractor([
                    make_dossier_evidence(module_file, module_text, section="3.2.P.8.1", score=0.82)]),
            )
            decision = decision_json(
                "gap" if is_gap else "no_gap", "value_inconsistency" if is_gap else "no_gap",
                severity, "strong", "3.2.P.8.1",
                (f"Shelf life differs between label ({label_months} months) and Module 3 ({module_months} months)."
                 if is_gap else f"Shelf life is consistent at {label_months} months."),
                (f"The label states '{label_text}', while retrieved Module 3 evidence states '{module_text}' — a difference of {diff} months."
                 if is_gap else f"Both the label and Module 3 evidence state the same shelf life of {label_months} months, despite different wording."),
                [guideline_citation("ICH_Q1A_R2", "Q1A_R2_Guideline.pdf", 10, "2.2", "ICH_Q1A_R2_2.2_chunk")],
                [dossier_citation(label_file, 1, None, f"{label_file}_chunk_1"),
                 dossier_citation(module_file, 1, "3.2.P.8.1", f"{module_file}_chunk_1")],
                "Align shelf-life information between Module 3 and the product label." if is_gap else "No immediate action required for shelf-life consistency.",
            )
            examples.append(make_chat_example(package, decision, "shelf_life"))
    return examples


# ---------------------------------------------------------------------------
# Category 3: storage condition — grid x wording variants
# ---------------------------------------------------------------------------
TEMP_VALUES = [20, 25, 30, 36]


def generate_storage_examples():
    examples = []
    for label_temp, module_temp in product(TEMP_VALUES, TEMP_VALUES):
        for _ in range(3):
            is_gap = label_temp != module_temp
            diff = abs(label_temp - module_temp)
            severity = "none" if not is_gap else ("medium" if diff <= 5 else "high")
            label_file = next_file(LABEL_FILE_POOL)
            module_file = next_file(MODULE_FILE_POOL)
            label_text = storage_phrase(label_temp)
            module_text = storage_phrase(module_temp)

            package = make_d2g_package(
                fact_type="storage_condition_statement", fact_text=label_text, source_file=label_file,
                guideline_evidence=make_guideline_evidence(
                    "Stability data should support the proposed storage condition.", "ICH_Q1A_R2", "2.2.7"),
                cross_dossier_evidence=maybe_with_distractor([
                    make_dossier_evidence(module_file, module_text, section="3.2.P.8.1", score=0.81)]),
            )
            decision = decision_json(
                "gap" if is_gap else "no_gap", "value_inconsistency" if is_gap else "no_gap",
                severity, "strong", "3.2.P.8.1",
                "Storage condition differs between label and Module 3." if is_gap else "Storage condition is consistent between label and Module 3.",
                (f"The label states '{label_text}', while Module 3 states '{module_text}'."
                 if is_gap else f"Both retrieved sources describe storage below {label_temp}°C, despite different wording."),
                [guideline_citation("ICH_Q1A_R2", "Q1A_R2_Guideline.pdf", 12, "2.2.7", "ICH_Q1A_R2_2.2.7_chunk")],
                [dossier_citation(label_file, 1, None, f"{label_file}_chunk_1"),
                 dossier_citation(module_file, 1, "3.2.P.8.1", f"{module_file}_chunk_1")],
                "Align storage condition information between Module 3 and the product label." if is_gap else "No immediate action required for storage-condition consistency.",
            )
            examples.append(make_chat_example(package, decision, "storage_condition"))
    return examples


# ---------------------------------------------------------------------------
# Category 4: batch identifiers — now actually uses product_name
# ---------------------------------------------------------------------------
BATCH_CASES = [
    ("B003", "Batch B001, Batch B002, and Batch B003 were placed on stability studies.", "no_gap"),
    ("B004", "Batch B001, Batch B002, and Batch B004 were placed on stability studies.", "no_gap"),
    ("B003", "Batch B001 and Batch B002 were placed on stability studies.", "gap"),
    ("B005", "Batch B001 and Batch B002 were placed on stability studies.", "gap"),
    ("B007", "Batch B006 was placed on stability studies.", "gap"),
    ("B003", "Batch B001 and Batch B002 were placed on stability studies; additional batches may be referenced elsewhere in the dossier.", "uncertain"),
    ("B010", "Several commercial-scale batches were evaluated during process validation.", "uncertain"),
]


def generate_batch_examples():
    examples = []
    for fact_batch, other_text, outcome in BATCH_CASES * 4:
        product_name = rand_product()
        module_file = next_file(MODULE_FILE_POOL)
        fact_text = f"For {product_name}, the primary commercial stability batch is {fact_batch}."
        cross_text = f"For {product_name}: {other_text}"

        package = make_d2g_package(
            fact_type="batch_identifier_statement", fact_text=fact_text, source_file="module_2_qos_variant.pdf",
            guideline_evidence=make_guideline_evidence(
                "The dossier should consistently identify batches used to support the stability and manufacturing data.",
                "ICH_M4Q", "3.2.P.5.4"),
            cross_dossier_evidence=maybe_with_distractor([
                make_dossier_evidence(module_file, cross_text, section="3.2.P.5.4",
                                      score=0.75 if outcome != "uncertain" else 0.55)]),
        )
        if outcome == "no_gap":
            decision = decision_json("no_gap", "no_gap", "none", "strong", "3.2.P.5.4",
                f"Batch {fact_batch} is corroborated by other dossier evidence.",
                f"The QOS names {fact_batch} as the primary stability batch for {product_name}, and Module 3 confirms it was included in stability studies.",
                [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 18, "3.2.P.5.4", "ICH_M4Q_3.2.P.5.4_chunk")],
                [dossier_citation("module_2_qos_variant.pdf", 1, None, "module_2_qos_variant.pdf_chunk_1"),
                 dossier_citation(module_file, 1, "3.2.P.5.4", f"{module_file}_chunk_1")],
                "No immediate action required for batch consistency.")
        elif outcome == "gap":
            decision = decision_json("gap", "value_inconsistency", "medium", "medium", "3.2.P.5.4",
                f"Batch {fact_batch} named in the QOS is not substantiated by other dossier evidence.",
                f"The QOS names {fact_batch} as the primary stability batch for {product_name}, but the retrieved Module 3 evidence only discusses different batches, with no mention of {fact_batch}.",
                [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 18, "3.2.P.5.4", "ICH_M4Q_3.2.P.5.4_chunk")],
                [dossier_citation("module_2_qos_variant.pdf", 1, None, "module_2_qos_variant.pdf_chunk_1"),
                 dossier_citation(module_file, 1, "3.2.P.5.4", f"{module_file}_chunk_1")],
                f"Verify whether {fact_batch} exists elsewhere in the dossier or correct the batch reference.")
        else:
            decision = decision_json("uncertain", "other", "low", "medium", "3.2.P.5.4",
                f"Evidence neither clearly confirms nor contradicts batch {fact_batch}.",
                f"The QOS names {fact_batch} for {product_name}, and the retrieved Module 3 evidence discusses related batch information without explicitly confirming or ruling out {fact_batch} — the evidence is ambiguous rather than contradictory.",
                [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 18, "3.2.P.5.4", "ICH_M4Q_3.2.P.5.4_chunk")],
                [dossier_citation("module_2_qos_variant.pdf", 1, None, "module_2_qos_variant.pdf_chunk_1"),
                 dossier_citation(module_file, 1, "3.2.P.5.4", f"{module_file}_chunk_1")],
                f"Clarify with the dossier author whether {fact_batch} is documented elsewhere.")
        examples.append(make_chat_example(package, decision, f"batch_{outcome}"))
    return examples


# ---------------------------------------------------------------------------
# Category 5: clinical endpoint mismatch
# ---------------------------------------------------------------------------
ENDPOINT_PAIRS = [
    ("change in fasting glucose at Week 12", "change in HbA1c at Week 12"),
    ("systolic blood pressure reduction at Week 12", "systolic blood pressure reduction at Week 12"),
    ("reduction in seizure frequency at Week 24", "reduction in seizure frequency at Week 24"),
    ("improvement in FEV1 at Week 8", "improvement in FVC at Week 8"),
    ("change in pain score at Week 6", "change in pain score at Week 6"),
    ("reduction in tumor size at Week 16", "overall survival at Week 16"),
    ("change in heart rate at Week 12", "change in heart rate at Week 12"),
    ("reduction in LDL cholesterol at Week 12", "reduction in triglycerides at Week 12"),
    ("improvement in disease activity score at Week 24", "improvement in disease activity score at Week 24"),
    ("time to symptom resolution", "time to symptom resolution"),
    ("change in bone mineral density at Week 52", "fracture incidence at Week 52"),
    ("reduction in relapse rate at Week 24", "reduction in relapse rate at Week 24"),
    ("change in renal function at Week 12", "change in proteinuria at Week 12"),
    ("improvement in quality of life score", "improvement in quality of life score"),
    ("reduction in migraine frequency at Week 12", "reduction in migraine severity at Week 12"),
]


def generate_endpoint_examples():
    examples = []
    for summary_endpoint, csr_endpoint in ENDPOINT_PAIRS * 3:
        is_gap = summary_endpoint != csr_endpoint
        summary_text = endpoint_intro(summary_endpoint)
        csr_text = endpoint_csr_intro(csr_endpoint)
        package = make_d2g_package(
            fact_type="clinical_endpoint_statement", fact_text=summary_text,
            source_file="module_2_clinical_summary_variant.pdf",
            guideline_evidence=make_guideline_evidence(
                "The clinical study report and clinical summary should clearly describe study endpoints and efficacy results.",
                "ICH_E3", "11.4"),
            cross_dossier_evidence=maybe_with_distractor([
                make_dossier_evidence("module_5_csr_variant.pdf", csr_text, section="7", score=0.84)]),
        )
        decision = decision_json(
            "gap" if is_gap else "no_gap", "endpoint_inconsistency" if is_gap else "no_gap",
            "high" if is_gap else "none", "strong", "7",
            "Primary endpoint differs between Clinical Summary and CSR." if is_gap else "Primary endpoint is consistent between Clinical Summary and CSR.",
            (f"The Clinical Summary states '{summary_endpoint}', while the CSR states '{csr_endpoint}' — these describe different measures, not the same endpoint worded differently."
             if is_gap else f"Both retrieved sources state '{summary_endpoint}'."),
            [guideline_citation("ICH_E3", "E3_Guideline.pdf", 20, "11.4", "ICH_E3_11.4_chunk")],
            [dossier_citation("module_2_clinical_summary_variant.pdf", 1, None, "module_2_clinical_summary_variant.pdf_chunk_1"),
             dossier_citation("module_5_csr_variant.pdf", 1, "7", "module_5_csr_variant.pdf_chunk_1")],
            "Verify the primary endpoint and align Clinical Summary with the CSR." if is_gap else "No immediate action required for endpoint consistency.",
        )
        examples.append(make_chat_example(package, decision, "endpoint"))
    return examples


# ---------------------------------------------------------------------------
# Category 6: SAE count mismatch — with wording variants
# ---------------------------------------------------------------------------
SAE_PAIRS = [
    (2, 6), (4, 4), (0, 0), (1, 3), (3, 3), (5, 9), (0, 2), (6, 6),
    (2, 2), (3, 7), (1, 1), (4, 8), (7, 7), (0, 1), (5, 5), (2, 4),
    (8, 8), (3, 5), (1, 4), (6, 10),
]


def generate_sae_examples():
    examples = []
    for summary_count, csr_count in SAE_PAIRS * 2:
        is_gap = summary_count != csr_count
        diff = abs(summary_count - csr_count)
        severity = "none" if not is_gap else ("high" if diff >= 2 else "medium")
        summary_text = sae_phrase(summary_count)
        csr_text = sae_phrase(csr_count)

        package = make_d2g_package(
            fact_type="safety_event_statement", fact_text=f"The clinical summary reports {summary_text}",
            source_file="module_2_clinical_summary_variant.pdf",
            guideline_evidence=make_guideline_evidence(
                "The clinical study report should include safety evaluation and serious adverse event information.",
                "ICH_E3", "12.3"),
            cross_dossier_evidence=maybe_with_distractor([
                make_dossier_evidence("module_5_csr_variant.pdf", csr_text, section="8", score=0.83)]),
        )
        decision = decision_json(
            "gap" if is_gap else "no_gap", "safety_count_inconsistency" if is_gap else "no_gap",
            severity, "strong", "8",
            "Serious adverse event count differs between Clinical Summary and CSR." if is_gap else "Serious adverse event count is consistent.",
            (f"The Clinical Summary reports '{summary_text}', while the CSR reports '{csr_text}' — a difference of {diff}."
             if is_gap else f"Both retrieved sources report {summary_count} serious adverse events, despite different wording."),
            [guideline_citation("ICH_E3", "E3_Guideline.pdf", 24, "12.3", "ICH_E3_12.3_chunk")],
            [dossier_citation("module_2_clinical_summary_variant.pdf", 1, None, "module_2_clinical_summary_variant.pdf_chunk_1"),
             dossier_citation("module_5_csr_variant.pdf", 1, "8", "module_5_csr_variant.pdf_chunk_1")],
            "Verify serious adverse event counts and align safety reporting." if is_gap else "No immediate action required for SAE count consistency.",
        )
        examples.append(make_chat_example(package, decision, "sae_count"))
    return examples


# ---------------------------------------------------------------------------
# Category 7: missing / present sections
# ---------------------------------------------------------------------------
MISSING_SECTIONS = [
    "3.2.P.8.2", "3.2.P.3.3", "5.3.5.1", "3.2.S.4.2", "2.7.4.2",
    "3.2.P.2.2.2", "3.2.A.2", "3.2.P.5.6", "2.7.1.2", "3.2.S.7.2",
    "3.2.P.4.4", "2.7.3.4", "3.2.S.2.4", "3.2.P.5.4", "3.2.A.3",
    "2.7.4.5", "3.2.S.1.2", "3.2.P.2.4",
]


def generate_missing_section_examples():
    examples = []
    for section in MISSING_SECTIONS * 4:
        nearby = section.rsplit(".", 1)[0] + ".1"
        requirement_text = f"{section} should be provided in the dossier, describing the relevant CTD content for this section."
        dossier_evidence = maybe_with_distractor([
            make_dossier_evidence("module_3_quality_stability_variant.pdf",
                f"The dossier contains nearby section {nearby} and related content, but no direct evidence of {section}.",
                section=nearby, score=0.60)])
        package = make_g2d_package("ICH_M4Q", "R1", "M4Q_R1_Guideline.pdf", section, requirement_text, dossier_evidence)
        decision = decision_json("gap", "missing_section", "high", "medium", section,
            f"Required section {section} is not directly evidenced in the retrieved dossier content.",
            f"The guideline expects {section}, but retrieved dossier evidence shows only nearby content ({nearby}) and no direct evidence of {section} itself.",
            [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 22, section, f"ICH_M4Q_{section}")],
            [dossier_citation("module_3_quality_stability_variant.pdf", 1, nearby, "module_3_quality_stability_variant.pdf_chunk_1")],
            f"Verify whether {section} exists elsewhere or add the missing required section.")
        examples.append(make_chat_example(package, decision, "missing_section"))
    return examples


NO_GAP_SECTIONS = ["3.2.P.8.1", "3.2.P.8.3", "3.2.P.3.1", "2.7.1", "2.7.4", "3.2.S.4.1", "3.2.P.1", "3.2.A.1", "2.7.3", "3.2.S.2.1"]


def generate_no_gap_section_examples():
    examples = []
    for section in NO_GAP_SECTIONS * 3:
        requirement_text = f"{section} should be provided in the dossier."
        dossier_evidence = [make_dossier_evidence("module_3_quality_stability_variant.pdf",
            f"{section} is present and contains content directly relevant to the requirement.",
            section=section, score=0.85)]
        package = make_g2d_package("ICH_M4Q", "R1", "M4Q_R1_Guideline.pdf", section, requirement_text, dossier_evidence)
        decision = decision_json("no_gap", "no_gap", "none", "strong", section,
            f"Retrieved dossier evidence directly addresses {section}.",
            f"The dossier evidence contains the same section {section} with directly relevant content.",
            [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 22, section, f"ICH_M4Q_{section}")],
            [dossier_citation("module_3_quality_stability_variant.pdf", 1, section, "module_3_quality_stability_variant.pdf_chunk_1")],
            "No immediate action required for this retrieved evidence package.")
        examples.append(make_chat_example(package, decision, "no_gap_section"))
    return examples


# ---------------------------------------------------------------------------
# Category 8: unsupported claims
# ---------------------------------------------------------------------------
CLAIM_CASES = [
    ("The product provides rapid symptom relief within 2 hours.",
     "The CSR reports Week 12 efficacy results but does not provide evidence for relief within 2 hours.", False),
    ("The product reduced the primary endpoint at Week 12 as described in the CSR.",
     "The CSR reports improvement in the primary endpoint at Week 12 consistent with the label claim.", True),
    ("The product is superior to standard of care.",
     "The CSR describes a placebo-controlled comparison and does not include an active comparator arm versus standard of care.", False),
    ("The product significantly reduced hospitalization rates.",
     "The CSR reports a statistically significant reduction in hospitalization rates during the study period.", True),
    ("The product is effective within the first dose.",
     "The CSR measures efficacy endpoints starting at Week 4, with no data on first-dose effects.", False),
    ("The product improved quality of life scores compared to baseline.",
     "The CSR reports a statistically significant improvement in quality of life scores from baseline.", True),
    ("The product eliminates all reported side effects seen with prior therapies.",
     "The CSR reports a reduced, but non-zero, rate of adverse events compared to prior therapies.", False),
    ("The product demonstrated a favorable safety profile consistent with the CSR.",
     "The CSR safety evaluation supports a favorable overall safety profile for the studied population.", True),
    ("The product provides 24-hour symptom control.",
     "The CSR only evaluated symptom scores at Week 12 and Week 24, without hourly or daily symptom tracking.", False),
    ("The product met its primary efficacy endpoint as described in the CSR.",
     "The CSR concludes the primary efficacy endpoint was met with statistical significance.", True),
    ("The product is proven to prevent disease progression in all patients.",
     "The CSR reports reduced progression in a subset of patients, not the full study population.", False),
    ("The product's safety data is consistent with the CSR safety evaluation.",
     "The CSR safety evaluation section reports adverse event data consistent with the label's safety summary.", True),
    ("The product works faster than all currently approved treatments.",
     "The CSR does not include a head-to-head comparison against other approved treatments.", False),
    ("The product's efficacy findings are consistent with the CSR's reported results.",
     "The CSR's efficacy results align with the summarized findings referenced in the label.", True),
    ("The product completely resolves the underlying condition.",
     "The CSR reports symptom improvement and disease control, not complete resolution of the underlying condition.", False),
    ("The product reduced relapse rates as shown in the CSR.",
     "The CSR reports a statistically significant reduction in relapse rates versus placebo.", True),
    ("The product is suitable for all patient populations without restriction.",
     "The CSR enrolled a specific adult population with defined inclusion and exclusion criteria.", False),
    ("The product's dosing regimen matches what is described in the CSR.",
     "The CSR describes the same dosing regimen referenced in the label.", True),
    ("The product has no drug interactions of clinical significance.",
     "The CSR does not include a dedicated drug-interaction study; interaction data is not addressed.", False),
    ("The product's reported benefit-risk profile matches the CSR discussion section.",
     "The CSR discussion section describes a benefit-risk profile consistent with the label's summary.", True),
]


def generate_claim_examples():
    examples = []
    for claim_text, cross_text, supported in CLAIM_CASES * 2:
        varied_claim_text = claim_intro(claim_text)
        package = make_d2g_package(
            fact_type="label_claim_statement", fact_text=varied_claim_text, source_file="label_variant.pdf",
            guideline_evidence=make_guideline_evidence(
                "Clinical claims should be supported by adequate clinical efficacy or safety evidence.",
                "ICH_M4E", "2.7.3"),
            cross_dossier_evidence=maybe_with_distractor([
                make_dossier_evidence("module_5_csr_variant.pdf", cross_text, section="7", score=0.79)]),
        )
        decision = decision_json(
            "no_gap" if supported else "gap", "no_gap" if supported else "unsupported_claim",
            "none" if supported else "medium", "medium", "7",
            "Label claim is supported by retrieved clinical evidence." if supported else "Label claim is not clearly supported by retrieved clinical evidence.",
            "The retrieved CSR evidence supports the label claim." if supported else "The label makes a claim broader or more specific than what the retrieved CSR evidence substantiates.",
            [guideline_citation("ICH_M4E", "M4E_R2_Guideline.pdf", 14, "2.7.3", "ICH_M4E_2.7.3_chunk")],
            [dossier_citation("label_variant.pdf", 1, None, "label_variant.pdf_chunk_1"),
             dossier_citation("module_5_csr_variant.pdf", 1, "7", "module_5_csr_variant.pdf_chunk_1")],
            "No immediate action required for this claim based on retrieved evidence." if supported else "Provide supporting clinical evidence for the claim or revise the label.",
        )
        examples.append(make_chat_example(package, decision, "unsupported_claim" if not supported else "supported_claim"))
    return examples


# ---------------------------------------------------------------------------
# Category 9: weak justification
# ---------------------------------------------------------------------------
WEAK_JUSTIFICATION_TEXTS = [
    "The justification is brief and does not clearly explain all stability data trends.",
    "The shelf-life rationale references limited long-term data without discussing observed trends.",
    "The storage condition justification does not address accelerated stability data trends.",
    "The stability conclusion is stated without discussing batch-to-batch variability.",
    "The justification does not explain how the proposed shelf life accounts for degradation trends.",
    "The rationale for the proposed specification limits is not clearly linked to the stability data presented.",
    "The justification references stability data without specifying the number of batches evaluated.",
    "The explanation for extrapolating shelf life beyond available data is not clearly provided.",
    "The justification does not address why accelerated and long-term data trends differ.",
    "The rationale for the proposed retest period is only briefly stated without supporting trend analysis.",
    "The justification for excluding certain stability parameters from evaluation is not explained.",
    "The stability conclusion does not discuss the statistical approach used to support the shelf-life claim.",
    "The justification references 'adequate stability' without quantifying supporting data.",
    "The explanation for the proposed shelf life does not address photostability considerations.",
    "The rationale for the specification range is stated briefly without linking to batch history.",
]


def generate_weak_justification_examples():
    examples = []
    for text in WEAK_JUSTIFICATION_TEXTS * 3:
        package = make_d2g_package(
            fact_type="shelf_life_statement",
            fact_text="The shelf life is justified based on the available stability data and supportive explanation.",
            source_file="module_3_quality_stability_variant.pdf",
            guideline_evidence=make_guideline_evidence(
                "Stability justification should be based on adequate stability data and evaluation of trends.",
                "ICH_Q1E", "2.5.1"),
            cross_dossier_evidence=maybe_with_distractor([
                make_dossier_evidence("module_3_quality_stability_variant.pdf", text, section="3.2.P.8.1", score=0.78)]),
        )
        decision = decision_json("gap", "weak_justification", "medium", "medium", "3.2.P.8.1",
            "Shelf-life justification appears weak based on retrieved evidence.",
            f"{text} Guideline evidence expects stability justification supported by data and trend evaluation, which the retrieved dossier text does not clearly demonstrate.",
            [guideline_citation("ICH_Q1E", "Q1E_Guideline.pdf", 10, "2.5.1", "ICH_Q1E_2.5.1_chunk")],
            [dossier_citation("module_3_quality_stability_variant.pdf", 1, "3.2.P.8.1", "module_3_quality_stability_variant.pdf_chunk_1")],
            "Review whether the proposed shelf life is adequately justified by stability data and trend analysis.")
        examples.append(make_chat_example(package, decision, "weak_justification"))
    return examples


# ---------------------------------------------------------------------------
# Category 10: uncertain
# ---------------------------------------------------------------------------
UNCERTAIN_CASES = [
    ("clinical_endpoint_statement", "The primary endpoint is described as improvement in patient-reported symptom burden.",
     "The CSR discusses several patient-reported measures without clearly identifying a single primary endpoint label."),
    ("manufacturer_statement", "The manufacturing site is described as the primary commercial facility.",
     "Module 3 references multiple manufacturing sites without clearly identifying which is the primary commercial facility."),
    ("shelf_life_statement", "The shelf life is proposed based on ongoing stability studies.",
     "Module 3 discusses ongoing stability studies without stating a specific concluded shelf-life value yet."),
    ("storage_condition_statement", "The storage condition is described as consistent with standard practice.",
     "Module 3 references general storage recommendations without specifying an exact temperature range."),
    ("batch_identifier_statement", "The validation batches are referenced as representative of commercial-scale production.",
     "Module 3 discusses process validation generally without listing specific batch numbers."),
    ("safety_event_statement", "The safety profile is described as generally consistent with the study population.",
     "The CSR discusses adverse events broadly without a clear total count in the retrieved excerpt."),
    ("clinical_endpoint_statement", "Secondary endpoints are described as supportive of the primary efficacy conclusion.",
     "The CSR lists several secondary measures without clearly distinguishing which supported the primary conclusion."),
    ("manufacturer_statement", "Contract manufacturing arrangements are referenced in the quality summary.",
     "Module 3 describes manufacturing responsibilities in general terms without naming a specific contract manufacturer."),
    ("label_claim_statement", "The product is described as generally well tolerated.",
     "The CSR discusses tolerability findings in a mixed manner without a clear overall tolerability conclusion in the retrieved excerpt."),
]


def generate_uncertain_examples():
    examples = []
    for fact_type, fact_text, cross_text in UNCERTAIN_CASES * 4:
        package = make_d2g_package(
            fact_type=fact_type, fact_text=fact_text, source_file="module_2_qos_variant.pdf",
            guideline_evidence=make_guideline_evidence(
                "The dossier should consistently and clearly describe the relevant regulatory fact across modules.",
                "ICH_M4Q", "3.2.P.5.4"),
            cross_dossier_evidence=maybe_with_distractor([
                make_dossier_evidence("module_3_manufacturing_variant.pdf", cross_text, section="3.2.P.5.4", score=0.55)
            ], chance=0.5),
        )
        decision = decision_json("uncertain", "other", "low", "medium", "3.2.P.5.4",
            "Retrieved evidence is topically relevant but does not clearly confirm or contradict the dossier fact.",
            f"{cross_text} This is related to the stated fact but does not directly confirm or explicitly contradict it, making a confident decision difficult without further clarification.",
            [guideline_citation("ICH_M4Q", "M4Q_R1_Guideline.pdf", 18, "3.2.P.5.4", "ICH_M4Q_3.2.P.5.4_chunk")],
            [dossier_citation("module_2_qos_variant.pdf", 1, None, "module_2_qos_variant.pdf_chunk_1"),
             dossier_citation("module_3_manufacturing_variant.pdf", 1, "3.2.P.5.4", "module_3_manufacturing_variant.pdf_chunk_1")],
            "Clarify the referenced value with the dossier author before final determination.")
        examples.append(make_chat_example(package, decision, "uncertain"))
    return examples


# ---------------------------------------------------------------------------
# Category 11: needs_human_review
# ---------------------------------------------------------------------------
HUMAN_REVIEW_CASES = [
    ("label_claim_statement", "The product has a favorable benefit profile.",
     "General clinical discussion is present, but direct support for the specific claim is unclear."),
    ("clinical_endpoint_statement", "The endpoint results are described as clinically meaningful.",
     "The retrieved excerpt is a table caption without accompanying descriptive text."),
    ("manufacturer_statement", "The manufacturer information is provided in the quality summary.",
     "The retrieved excerpt is truncated and does not contain a clear manufacturer statement."),
    ("shelf_life_statement", "The shelf life is described in the stability section.",
     "The retrieved excerpt references a cross-reference to another section without the actual value."),
    ("batch_identifier_statement", "Batch information is referenced in the manufacturing section.",
     "The retrieved excerpt only contains a section heading with no batch details."),
    ("safety_event_statement", "Safety information is summarized in the CSR.",
     "The retrieved excerpt is an unrelated administrative statement about document formatting."),
    ("storage_condition_statement", "Storage conditions are described in the stability section.",
     "The retrieved excerpt does not mention any temperature or storage details."),
    ("label_claim_statement", "The label describes the product's intended use.",
     "The retrieved excerpt is a boilerplate disclaimer unrelated to the specific claim."),
    ("clinical_endpoint_statement", "The primary endpoint is defined in the protocol synopsis.",
     "No protocol synopsis content was retrieved; only an unrelated administrative section was found."),
]


def generate_human_review_examples():
    examples = []
    for fact_type, fact_text, cross_text in HUMAN_REVIEW_CASES * 4:
        package = make_d2g_package(
            fact_type=fact_type, fact_text=fact_text, source_file="label_variant.pdf",
            guideline_evidence=[],
            cross_dossier_evidence=[make_dossier_evidence("module_5_csr_variant.pdf", cross_text, section="9", score=0.30)],
        )
        package["guideline_evidence_strength"] = "missing"
        package["cross_dossier_evidence_strength"] = "weak"
        decision = decision_json("needs_human_review", "insufficient_evidence", "medium", "weak", "9",
            "Retrieved evidence is insufficient to make a confident determination.",
            f"Guideline evidence is missing and the retrieved dossier evidence ('{cross_text}') does not provide clear support one way or the other, so this requires human review.",
            [], [dossier_citation("module_5_csr_variant.pdf", 1, "9", "module_5_csr_variant.pdf_chunk_1")],
            "Manually review the fact and supporting evidence before making a final determination.")
        examples.append(make_chat_example(package, decision, "needs_human_review"))
    return examples


# ---------------------------------------------------------------------------
# Validation + reporting
# ---------------------------------------------------------------------------
def validate_examples(examples: list[dict]) -> None:
    errors = []
    for i, example in enumerate(examples):
        decision = json.loads(example["messages"][2]["content"])
        if decision.get("llm_decision") not in ALLOWED_DECISIONS:
            errors.append(f"example {i}: invalid llm_decision {decision.get('llm_decision')!r}")
        if decision.get("finding_type") not in ALLOWED_FINDING_TYPES:
            errors.append(f"example {i}: invalid finding_type {decision.get('finding_type')!r}")
        if decision.get("severity") not in ALLOWED_SEVERITIES:
            errors.append(f"example {i}: invalid severity {decision.get('severity')!r}")
        if decision.get("evidence_status") not in ALLOWED_EVIDENCE_STATUS:
            errors.append(f"example {i}: invalid evidence_status {decision.get('evidence_status')!r}")
        # Enforce the exact consistency rule from the feedback: certain
        # finding_types should never co-occur with llm_decision == "no_gap".
        gap_only_findings = {
            "missing_section", "value_inconsistency", "entity_inconsistency",
            "endpoint_inconsistency", "safety_count_inconsistency",
            "unsupported_claim", "weak_justification",
        }
        if decision.get("finding_type") in gap_only_findings and decision.get("llm_decision") != "gap":
            errors.append(
                f"example {i}: finding_type={decision.get('finding_type')!r} "
                f"but llm_decision={decision.get('llm_decision')!r} (should be 'gap')"
            )
    if errors:
        raise ValueError("Schema validation failed:\n" + "\n".join(errors))
    print(f"Schema validation passed for all {len(examples)} examples.")


def check_no_demo_leakage(examples: list[dict]) -> None:
    """Confirm nothing from the actual demo dossier (product name
    'Cardiostat', or its exact planted values) leaked into training data."""
    banned_terms = ["Cardiostat", "Cardionil"]
    hits = []
    for i, example in enumerate(examples):
        full_text = json.dumps(example)
        for term in banned_terms:
            if term in full_text:
                hits.append((i, term))
    if hits:
        raise ValueError(f"Demo dossier leakage detected: {hits}")
    print("No demo dossier leakage detected (checked for: " + ", ".join(banned_terms) + ").")


def print_label_distribution(examples: list[dict]) -> None:
    decision_counts, severity_counts, finding_counts = {}, {}, {}
    for example in examples:
        decision = json.loads(example["messages"][2]["content"])
        decision_counts[decision["llm_decision"]] = decision_counts.get(decision["llm_decision"], 0) + 1
        severity_counts[decision["severity"]] = severity_counts.get(decision["severity"], 0) + 1
        finding_counts[decision["finding_type"]] = finding_counts.get(decision["finding_type"], 0) + 1

    print("\nllm_decision distribution:")
    for key in sorted(ALLOWED_DECISIONS):
        pct = 100 * decision_counts.get(key, 0) / len(examples)
        print(f"  {key:20} | {decision_counts.get(key, 0):4}  ({pct:.1f}%)")

    print("\nseverity distribution:")
    for key in sorted(ALLOWED_SEVERITIES):
        print(f"  {key:20} | {severity_counts.get(key, 0)}")

    print("\nfinding_type distribution:")
    for key in sorted(ALLOWED_FINDING_TYPES):
        print(f"  {key:28} | {finding_counts.get(key, 0)}")


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(total, train, eval_, test, path: Path) -> None:
    manifest = {
        "dataset_version": DATASET_VERSION,
        "total_examples": total,
        "train_examples": train,
        "eval_examples": eval_,
        "test_examples": test,
        "created_for": "LLM regulatory dossier gap decision layer",
        "not_used": "gold_issues.csv",
        "notes": (
            "Synthetic data only. No values, product names, or documents from "
            "sample_dossier/ or eval/gold_issues.csv are used — the final demo "
            "dossier is held out entirely and never seen during training."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def generate_examples() -> list[dict]:
    examples = []
    examples += generate_manufacturer_examples()
    examples += generate_shelf_life_examples()
    examples += generate_storage_examples()
    examples += generate_batch_examples()
    examples += generate_endpoint_examples()
    examples += generate_sae_examples()
    examples += generate_missing_section_examples()
    examples += generate_no_gap_section_examples()
    examples += generate_claim_examples()
    examples += generate_weak_justification_examples()
    examples += generate_uncertain_examples()
    examples += generate_human_review_examples()
    return examples


def main() -> None:
    random.seed(RANDOM_SEED)

    examples = generate_examples()
    validate_examples(examples)
    check_no_demo_leakage(examples)
    print_label_distribution(examples)

    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * 0.8)
    eval_end = int(n * 0.9)

    train_records = examples[:train_end]
    eval_records = examples[train_end:eval_end]
    test_records = examples[eval_end:]  # held out, never used in training

    write_jsonl(train_records, TRAIN_PATH)
    write_jsonl(eval_records, EVAL_PATH)
    write_jsonl(test_records, TEST_PATH)
    write_manifest(n, len(train_records), len(eval_records), len(test_records), MANIFEST_PATH)

    print("\nSFT Training Data Generation Summary (v2)")
    print("------------------------------------------")
    print(f"Total examples : {n}")
    print(f"Train examples : {len(train_records)}")
    print(f"Eval examples  : {len(eval_records)}")
    print(f"Test examples  : {len(test_records)} (held out — do not train on this)")
    print(f"Train path     : {TRAIN_PATH}")
    print(f"Eval path      : {EVAL_PATH}")
    print(f"Test path      : {TEST_PATH}")
    print(f"Manifest       : {MANIFEST_PATH}")


if __name__ == "__main__":
    main()