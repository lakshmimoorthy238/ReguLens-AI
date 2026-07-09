# Project Scope

## Project Title
LLM-First RAG-Based Regulatory Dossier Gap Assistant with Rule-Based Verification

## Problem Statement
Regulatory dossier review requires checking whether CTD-style pharmaceutical submission documents are complete, consistent, and traceable to regulatory expectations. Manual review is time-consuming and error-prone.

## Objective
Build a prototype system that uses dual RAG and a fine-tuned LLM to flag potential gaps and inconsistencies in CTD-style pharmaceutical dossiers.

## Core AI Scope
The fine-tuned LLM acts as the primary decision layer. It performs:
- CTD section classification
- Regulatory entity extraction
- Regulatory guideline understanding using RAG
- Gap detection
- Severity assignment
- Reviewer action generation

A rule-based verification layer double-checks deterministic findings such as:
- missing sections
- value mismatches
- evidence presence
- JSON validity
- severity consistency

## MVP Product Type
Small-molecule immediate-release oral tablet.

## Supported Documents
- Module 2.3 Quality Overall Summary
- Module 3 Quality / Stability
- Module 3 Manufacturing
- Module 4 Nonclinical Summary
- Module 5 Clinical Study Report
- Product Label

## Supported Gap Types
1. Missing 3.2.P.8.2 stability protocol
2. Shelf-life mismatch
3. Storage-condition mismatch
4. Batch mismatch
5. Manufacturer mismatch
6. Primary endpoint mismatch
7. Serious adverse event count mismatch
8. CSR synopsis missing
9. Unsupported claim
10. Weak justification

## Out of Scope
- Full regulatory compliance certification
- FDA/eCTD XML validation
- Drug approval/rejection decisions
- Full clinical statistical review
- All product types
- All global regions
- Production security certification

## Final Output
Evidence-backed dossier gap report with guideline citations, dossier citations, severity, status, and reviewer action.