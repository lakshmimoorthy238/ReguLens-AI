from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "sample_dossier"


def draw_wrapped_text(c, text, x, y, max_width=90, line_height=14):
    """
    Simple text wrapper for PDF generation.
    max_width is approximate character count, not PDF units.
    """
    lines = []

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()

        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()

            if len(test_line) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

    for line in lines:
        if y < inch:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = letter[1] - inch

        c.drawString(x, y, line)
        y -= line_height

    return y


def create_pdf(file_name: str, title: str, body: str):
    """Create a simple synthetic PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / file_name
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, height - inch, title)

    c.setFont("Helvetica", 10)
    y = height - 1.4 * inch

    draw_wrapped_text(c, body, inch, y)

    c.save()
    print(f"Created: {pdf_path}")


def main():
    module_2_qos = """
2.3 Quality Overall Summary

Product Name: Cardiostat 10 mg film-coated tablets
Active Substance: Cardionil hydrochloride
Dosage Form: Film-coated tablet
Route of Administration: Oral
Strength: 10 mg

2.3.P Drug Product Summary

The drug product is an immediate-release oral tablet intended for the treatment of hypertension.

The Quality Overall Summary states that the primary commercial stability batch is B003.

Manufacturer listed in the Quality Overall Summary:
Apex Pharma Manufacturing Pvt. Ltd.

The proposed shelf life in the quality summary is 24 months when stored below 25°C.

The Quality Overall Summary refers to Module 3 for detailed manufacturing and stability data.

Reviewer Note:
The QOS should be consistent with the detailed Module 3 quality documentation.
"""

    module_2_clinical_summary = """
2.7 Clinical Summary

Product Name: Cardiostat 10 mg film-coated tablets

2.7.3 Summary of Clinical Efficacy

Study ID: CSR-CARDIO-101

The primary efficacy endpoint in the clinical summary is reduction in systolic blood pressure from baseline to Week 12.

The study population included adult patients with mild to moderate hypertension.

The clinical summary states that the treatment demonstrated clinically meaningful reduction in blood pressure.

2.7.4 Summary of Clinical Safety

The clinical summary reports 3 serious adverse events during the treatment period.

No deaths were reported.

Reviewer Note:
The Clinical Summary should be consistent with the full Clinical Study Report in Module 5.
"""

    module_3_stability = """
3.2.P Drug Product

3.2.P.1 Description and Composition of the Drug Product

Cardiostat 10 mg film-coated tablets contain Cardionil hydrochloride as the active substance.

3.2.P.8 Stability

3.2.P.8.1 Stability Summary and Conclusions

The proposed shelf life of Cardiostat 10 mg film-coated tablets is 24 months.

The recommended storage condition is: Store below 25°C.

The shelf life is justified based on limited available long-term stability data and supportive accelerated stability data.

The stability conclusion states that the drug product remains within proposed specifications during the available study period.

Note:
The justification for the proposed 24-month shelf life is brief and does not clearly explain all data trends.

3.2.P.8.3 Stability Data

Batch B001 and Batch B002 were placed on long-term and accelerated stability studies.

Available long-term data support continued monitoring of assay, degradation products, dissolution, and appearance.

Stability data tables are summarized in this section.
"""

    module_3_manufacturing = """
3.2.P.3 Manufacture

Product Name: Cardiostat 10 mg film-coated tablets

3.2.P.3.1 Manufacturer

The drug product manufacturer is:
Nova Labs Manufacturing Pvt. Ltd.

Manufacturing Site:
Nova Labs Manufacturing Pvt. Ltd., Hyderabad, India.

3.2.P.3.3 Description of Manufacturing Process and Process Controls

The drug product is manufactured using wet granulation, compression, and film coating.

Commercial-scale validation batches:
B001
B002

The manufacturing process includes dispensing, granulation, drying, blending, compression, coating, and packaging.

Reviewer Note:
Manufacturing details should be consistent with the Quality Overall Summary.
"""

    module_4_nonclinical = """
Module 4 Nonclinical Study Reports

4.1 Table of Contents of Module 4

4.2 Study Reports

4.2.1 Pharmacology

A primary pharmacodynamic study was conducted to evaluate blood pressure lowering activity in an animal model.

4.2.2 Pharmacokinetics

Basic pharmacokinetic data are summarized for systemic exposure.

4.2.3 Toxicology

A repeat-dose toxicity study was conducted in rodents.

No major unexpected toxicological findings were reported in this synthetic summary.

Reviewer Note:
This MVP performs only high-level Module 4 structure mapping and does not judge scientific adequacy of nonclinical studies.
"""

    module_5_csr = """
Module 5 Clinical Study Report

Study ID: CSR-CARDIO-101
Study Title: A randomized controlled study of Cardiostat 10 mg tablets in adult patients with hypertension.

1. Ethics

The study was conducted in accordance with ethical principles and informed consent requirements.

2. Investigators and Study Administrative Structure

The study was conducted at multiple clinical sites.

3. Introduction

Cardiostat is being developed for treatment of hypertension.

4. Study Objectives

The primary objective of the study was to evaluate the effect of Cardiostat on heart rate reduction from baseline to Week 12.

5. Investigational Plan

The study used a randomized, double-blind, parallel-group design.

6. Study Patients

Adult patients with mild to moderate hypertension were enrolled.

7. Efficacy Evaluation

The primary endpoint reported in this Clinical Study Report is heart rate reduction from baseline to Week 12.

8. Safety Evaluation

A total of 5 serious adverse events were reported during the treatment period.

No deaths were reported.

9. Discussion and Conclusions

The study suggests potential clinical benefit. Further review is required to align endpoint and safety reporting with the Clinical Summary.
"""

    label = """
Product Label

Product Name: Cardiostat 10 mg film-coated tablets
Active Substance: Cardionil hydrochloride
Route: Oral administration

Indication:
Cardiostat is indicated for the treatment of hypertension in adults.

Shelf Life:
18 months.

Storage Condition:
Store below 30°C.

Claim:
Cardiostat provides rapid control of blood pressure within 24 hours.

Reviewer Note:
The rapid control claim should be supported by clinical evidence in the dossier.
"""

    create_pdf("module_2_qos.pdf", "Module 2.3 Quality Overall Summary", module_2_qos)
    create_pdf("module_2_clinical_summary.pdf", "Module 2.7 Clinical Summary", module_2_clinical_summary)
    create_pdf("module_3_quality_stability.pdf", "Module 3 Quality - Stability", module_3_stability)
    create_pdf("module_3_manufacturing.pdf", "Module 3 Quality - Manufacturing", module_3_manufacturing)
    create_pdf("module_4_nonclinical_summary.pdf", "Module 4 Nonclinical Summary", module_4_nonclinical)
    create_pdf("module_5_clinical_study_report.pdf", "Module 5 Clinical Study Report", module_5_csr)
    create_pdf("label.pdf", "Product Label", label)


if __name__ == "__main__":
    main()python scripts\generate_sample_dossier.py