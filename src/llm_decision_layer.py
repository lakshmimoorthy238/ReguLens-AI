import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "outputs" / "bidirectional_evidence_packages.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "llm_decisions.json"

# NOTE: this is a prompted baseline model, not yet the fine-tuned
# Llama-3.1-8B / Mistral-7B checkpoint from the project plan. It exists so
# the decision layer can be built and tested end-to-end now; swap in a
# fine-tuned model path via --model once LoRA fine-tuning is done.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_EVIDENCE_ITEMS = 3
MAX_TEXT_CHARS = 900

# How often (in packages) to write a checkpoint to disk during a long run,
# so an interruption doesn't lose everything decided so far.
CHECKPOINT_EVERY = 10


ALLOWED_DECISIONS = {
    "gap",
    "no_gap",
    "uncertain",
    "needs_human_review",
}

ALLOWED_SEVERITIES = {
    "high",
    "medium",
    "low",
    "none",
}

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

Return exactly this JSON schema:
{
  "llm_decision": "gap | no_gap | uncertain | needs_human_review",
  "finding_type": "missing_section | value_inconsistency | entity_inconsistency | endpoint_inconsistency | safety_count_inconsistency | unsupported_claim | weak_justification | insufficient_evidence | no_gap | other",
  "severity": "high | medium | low | none",
  "evidence_status": "strong | medium | weak | missing",
  "ctd_section": "string or null",
  "finding_summary": "one sentence summary",
  "reasoning_summary": "brief evidence-based reason",
  "guideline_citations": [
    {
      "guideline_id": "string",
      "source_file": "string",
      "page_number": "number or null",
      "section": "string or null",
      "chunk_id": "string"
    }
  ],
  "dossier_citations": [
    {
      "file_name": "string",
      "page_number": "number or null",
      "section": "string or null",
      "chunk_id": "string"
    }
  ],
  "reviewer_action": "specific action for human reviewer",
  "needs_rule_verification": true
}
"""


def load_input_packages(path: Path = INPUT_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Evidence package file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "packages" in payload:
        return payload["packages"]

    if isinstance(payload, list):
        return payload

    raise ValueError("Input file must contain either a list or a dict with key 'packages'.")


def save_output(decisions: list[dict], path: Path = OUTPUT_PATH, partial: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_type": "llm_decision_layer",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_PATH),
        "partial_checkpoint": partial,
        "note": (
            "These are LLM-generated decisions from neutral bidirectional evidence packages. "
            "gold_issues.csv is not used in this step."
        ),
        "decision_count": len(decisions),
        "decisions": decisions,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def trim_text(text: Optional[str], max_chars: int = MAX_TEXT_CHARS) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text)).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + " ...[truncated]"


def select_diverse_evidence(
    items: list[dict],
    max_items: int,
    key_fn,
) -> list[dict]:
    """
    Truncate an evidence list to max_items, but prioritize covering
    distinct sources (by key_fn, e.g. file_name) before adding a second
    item from a source already included.

    Without this, plain top-N-by-score truncation can silently drop the
    one source that actually contradicts another (e.g. a label.pdf chunk
    with "18 months" losing out to three near-duplicate module_3 chunks
    all saying "24 months") — which would hide the exact mismatch this
    system exists to catch, not because retrieval failed, but because
    truncation discarded the evidence.
    """
    if len(items) <= max_items:
        return items

    selected = []
    seen_keys = set()

    for item in items:
        key = key_fn(item)
        if key not in seen_keys:
            selected.append(item)
            seen_keys.add(key)
        if len(selected) >= max_items:
            return selected

    for item in items:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= max_items:
            break

    return selected


def compact_guideline_item(item: dict) -> dict:
    return {
        "retrieval_score": item.get("retrieval_score"),
        "evidence_status": item.get("evidence_status"),
        "guideline_id": item.get("guideline_id"),
        "title": item.get("title"),
        "version": item.get("version"),
        "domain": item.get("domain"),
        "source_file": item.get("source_file"),
        "page_number": item.get("page_number"),
        "section": item.get("detected_section") or item.get("section"),
        "chunk_id": item.get("chunk_id"),
        "text": trim_text(item.get("text")),
    }


def compact_dossier_item(item: dict) -> dict:
    return {
        "retrieval_score": item.get("retrieval_score"),
        "evidence_status": item.get("evidence_status"),
        "file_name": item.get("file_name"),
        "document_type": item.get("document_type"),
        "module_guess": item.get("module_guess"),
        "page_number": item.get("page_number"),
        "section": item.get("detected_section"),
        "chunk_id": item.get("chunk_id"),
        "source_hash": item.get("source_hash"),
        "text": trim_text(item.get("text")),
    }


def compact_package(package: dict, package_id: str) -> dict:
    direction = package.get("direction")

    if direction == "guideline_to_dossier":
        dossier_evidence = select_diverse_evidence(
            package.get("dossier_evidence", []),
            MAX_EVIDENCE_ITEMS,
            key_fn=lambda item: item.get("file_name"),
        )

        return {
            "package_id": package_id,
            "direction": "guideline_to_dossier",
            "source_type": package.get("source_type"),
            # Ground-truth identifiers carried through independent of
            # whatever the LLM itself reports — used for traceability in
            # the final decision record, not just embedded in the prompt.
            "_trace": {
                "guideline_id": package.get("guideline_id"),
                "section": package.get("section"),
            },
            "guideline_requirement": {
                "guideline_id": package.get("guideline_id"),
                "guideline_version": package.get("guideline_version"),
                "section": package.get("section"),
                "title": package.get("guideline_title"),
                "domain": package.get("guideline_domain"),
                "source_file": package.get("guideline_source_file"),
                "page_number": package.get("guideline_page_number"),
                "chunk_id": package.get("guideline_chunk_id"),
                "requirement_text": trim_text(package.get("guideline_requirement_text")),
            },
            "dossier_evidence_strength": package.get("dossier_evidence_strength"),
            "dossier_evidence": [compact_dossier_item(item) for item in dossier_evidence],
            "task": package.get("llm_task"),
        }

    if direction == "dossier_to_guideline":
        fact = package.get("fact", {})
        source = fact.get("source", {})

        guideline_evidence = select_diverse_evidence(
            package.get("guideline_evidence", []),
            MAX_EVIDENCE_ITEMS,
            key_fn=lambda item: item.get("guideline_id"),
        )
        cross_dossier_evidence = select_diverse_evidence(
            package.get("cross_dossier_evidence", []),
            MAX_EVIDENCE_ITEMS,
            key_fn=lambda item: item.get("file_name"),
        )

        return {
            "package_id": package_id,
            "direction": "dossier_to_guideline",
            "source_type": package.get("source_type"),
            "_trace": {
                "fact_id": fact.get("fact_id"),
                "fact_type": fact.get("fact_type"),
                "source_file": source.get("file_name"),
                "section": source.get("detected_section"),
            },
            "dossier_fact": {
                "fact_id": fact.get("fact_id"),
                "fact_type": fact.get("fact_type"),
                "extracted_value": fact.get("extracted_value"),
                "fact_text": trim_text(fact.get("fact_text")),
                "source": {
                    "file_name": source.get("file_name"),
                    "document_type": source.get("document_type"),
                    "module_guess": source.get("module_guess"),
                    "page_number": source.get("page_number"),
                    "section": source.get("detected_section"),
                    "chunk_id": source.get("chunk_id"),
                    "source_hash": source.get("source_hash"),
                },
            },
            "guideline_evidence_strength": package.get("guideline_evidence_strength"),
            "guideline_evidence": [compact_guideline_item(item) for item in guideline_evidence],
            "cross_dossier_evidence_strength": package.get("cross_dossier_evidence_strength"),
            "cross_dossier_evidence": [compact_dossier_item(item) for item in cross_dossier_evidence],
            "task": package.get("llm_task"),
        }

    return {
        "package_id": package_id,
        "direction": direction,
        "_trace": {},
        "raw_package": package,
    }


def build_user_prompt(compact: dict) -> str:
    # Strip the internal _trace block before it goes to the model — it's
    # bookkeeping for us, not something the LLM needs to see or reason over.
    prompt_view = {k: v for k, v in compact.items() if k != "_trace"}

    return (
        "Review the following evidence package and return only valid JSON using the required schema.\n\n"
        "Evidence package:\n"
        f"{json.dumps(prompt_view, indent=2, ensure_ascii=False)}"
    )


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading LLM model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.to(device)

    model.eval()

    return tokenizer, model, device


def make_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"System:\n{SYSTEM_PROMPT.strip()}\n\n"
        f"User:\n{user_prompt.strip()}\n\n"
        f"Assistant:\n"
    )


def generate_response(
    tokenizer,
    model,
    device: str,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def extract_first_json_object(text: str) -> dict:
    """
    Extract the first JSON object from model output.
    Handles cases where the model adds small text before/after JSON.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object start found in model output.")

    brace_count = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1

            if brace_count == 0:
                json_text = text[start:idx + 1]
                return json.loads(json_text)

    raise ValueError("Could not find complete JSON object in model output.")


def infer_default_evidence_status(compact: dict) -> str:
    direction = compact.get("direction")

    if direction == "guideline_to_dossier":
        return compact.get("dossier_evidence_strength") or "weak"

    if direction == "dossier_to_guideline":
        guideline_strength = compact.get("guideline_evidence_strength") or "weak"
        cross_strength = compact.get("cross_dossier_evidence_strength") or "weak"

        if "missing" in {guideline_strength, cross_strength}:
            return "missing"
        if "weak" in {guideline_strength, cross_strength}:
            return "weak"
        if "medium" in {guideline_strength, cross_strength}:
            return "medium"
        return "strong"

    return "weak"


def normalize_decision(
    parsed: dict,
    compact: dict,
    package_id: str,
    raw_response: str,
) -> dict:
    decision = parsed.get("llm_decision", "uncertain")
    finding_type = parsed.get("finding_type", "other")
    severity = parsed.get("severity", "medium")
    evidence_status = parsed.get("evidence_status") or infer_default_evidence_status(compact)

    if decision not in ALLOWED_DECISIONS:
        decision = "uncertain"

    if finding_type not in ALLOWED_FINDING_TYPES:
        finding_type = "other"

    if severity not in ALLOWED_SEVERITIES:
        severity = "medium"

    if decision == "no_gap":
        severity = "none"
        if finding_type not in {"no_gap", "other"}:
            finding_type = "no_gap"

    if evidence_status not in {"strong", "medium", "weak", "missing"}:
        evidence_status = infer_default_evidence_status(compact)

    return {
        "package_id": package_id,
        "direction": compact.get("direction"),
        "source_type": compact.get("source_type"),
        "trace": compact.get("_trace", {}),
        "llm_decision": decision,
        "finding_type": finding_type,
        "severity": severity,
        "evidence_status": evidence_status,
        "ctd_section": parsed.get("ctd_section"),
        "finding_summary": parsed.get("finding_summary", ""),
        "reasoning_summary": parsed.get("reasoning_summary", ""),
        "guideline_citations": parsed.get("guideline_citations", []),
        "dossier_citations": parsed.get("dossier_citations", []),
        "reviewer_action": parsed.get("reviewer_action", ""),
        "needs_rule_verification": bool(parsed.get("needs_rule_verification", True)),
        "raw_model_response": raw_response,
    }


def make_error_decision(
    compact: dict,
    package_id: str,
    raw_response: str,
    error: str,
) -> dict:
    return {
        "package_id": package_id,
        "direction": compact.get("direction"),
        "source_type": compact.get("source_type"),
        "trace": compact.get("_trace", {}),
        "llm_decision": "needs_human_review",
        "finding_type": "insufficient_evidence",
        "severity": "medium",
        "evidence_status": infer_default_evidence_status(compact),
        "ctd_section": None,
        "finding_summary": "Model output could not be generated or parsed.",
        "reasoning_summary": f"Error: {error}",
        "guideline_citations": [],
        "dossier_citations": [],
        "reviewer_action": "Review this evidence package manually because the model step failed.",
        "needs_rule_verification": True,
        "raw_model_response": raw_response,
    }


def package_passes_filters(
    package: dict,
    direction: Optional[str],
) -> bool:
    if direction and package.get("direction") != direction:
        return False

    return True


def run_decision_layer(
    packages: list[dict],
    tokenizer,
    model,
    device: str,
    direction: Optional[str],
    max_packages: Optional[int],
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[dict]:
    selected = [
        package for package in packages
        if package_passes_filters(package, direction)
    ]

    if max_packages:
        selected = selected[:max_packages]

    print(f"Packages selected for LLM decision: {len(selected)}")

    decisions = []

    try:
        for idx, package in enumerate(selected, start=1):
            package_id = f"PKG-{idx:05d}"

            print(f"LLM deciding {package_id} | direction={package.get('direction')}")

            compact = compact_package(package, package_id)
            user_prompt = build_user_prompt(compact)
            prompt = make_chat_prompt(tokenizer, user_prompt)

            raw_response = ""
            try:
                raw_response = generate_response(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    prompt=prompt,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                )
                parsed = extract_first_json_object(raw_response)
                decision = normalize_decision(
                    parsed=parsed,
                    compact=compact,
                    package_id=package_id,
                    raw_response=raw_response,
                )
            except Exception as exc:
                print(f"  FAILED on {package_id}: {exc}")
                decision = make_error_decision(
                    compact=compact,
                    package_id=package_id,
                    raw_response=raw_response,
                    error=str(exc),
                )

            decisions.append(decision)

            if CHECKPOINT_EVERY and idx % CHECKPOINT_EVERY == 0:
                save_output(decisions, partial=True)
                print(f"  Checkpoint saved ({idx}/{len(selected)} decisions so far).")
    finally:
        # Ensures partial progress survives a crash or Ctrl+C, not just a
        # clean finish — the caller still does a final, non-partial save.
        if decisions:
            save_output(decisions, partial=True)

    return decisions


def print_summary(decisions: list[dict]) -> None:
    print("\nLLM Decision Layer Summary")
    print("--------------------------")
    print(f"Decisions created : {len(decisions)}")
    print(f"Output file       : {OUTPUT_PATH}")
    print()

    counts = {}

    for decision in decisions:
        key = decision.get("llm_decision", "unknown")
        counts[key] = counts.get(key, 0) + 1

    for key, count in sorted(counts.items()):
        print(f"{key:20} | {count:4}")

    print()

    finding_counts = {}
    for decision in decisions:
        key = decision.get("finding_type", "unknown")
        finding_counts[key] = finding_counts.get(key, 0) + 1

    for key, count in sorted(finding_counts.items()):
        print(f"{key:35} | {count:4}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM decision layer for regulatory gap detection")

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name",
    )

    parser.add_argument(
        "--direction",
        choices=["guideline_to_dossier", "dossier_to_guideline"],
        default=None,
        help="Optionally process only one direction",
    )

    parser.add_argument(
        "--max-packages",
        type=int,
        default=5,
        help="Limit packages for testing. Use a larger number after testing.",
    )

    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=3500,
        help="Maximum prompt tokens sent to the model",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=700,
        help="Maximum output tokens generated by the model",
    )

    args = parser.parse_args()

    packages = load_input_packages(INPUT_PATH)

    tokenizer, model, device = load_model(args.model)

    decisions = run_decision_layer(
        packages=packages,
        tokenizer=tokenizer,
        model=model,
        device=device,
        direction=args.direction,
        max_packages=args.max_packages,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    save_output(decisions, partial=False)
    print_summary(decisions)


if __name__ == "__main__":
    main()