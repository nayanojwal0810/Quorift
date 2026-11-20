"""Build 20 active banking policy PDFs and the document registry for Quorift."""

import hashlib
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from quorift.config import paths


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header for subsequent pages
        if self._pageNumber > 1:
            self.drawString(54, 750, "Apex Crest Bank — Internal Policy | Fictional Demonstration Document")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)

        self.drawString(54, 32, "CONFIDENTIAL & PROPRIETARY — APEX CREST BANK DEMONSTRATION REPOSITORY")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


def compute_source_hash(policy: dict) -> str:
    """Compute SHA-256 over a normalized representation of source content."""
    parts = [
        policy["document_id"].strip().upper(),
        policy["version"].strip(),
        policy["title"].strip(),
        policy["department"].strip(),
        policy["classification"].strip(),
        ",".join(sorted(policy["allowed_roles"])),
    ]
    for s in policy["sections"]:
        parts.append(
            f"{s['section_number'].strip()}|{s['heading'].strip()}|{s['content'].strip()}"
        )
    normalized_str = "\n".join(parts)
    return hashlib.sha256(normalized_str.encode("utf-8")).hexdigest()


def generate_pdf(policy: dict, out_pdf_path: Path) -> None:
    """Render a single policy document into a styled, accessible PDF."""
    doc = SimpleDocTemplate(
        str(out_pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F294A"),
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
    )
    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#718096"),
        spaceAfter=12,
    )
    meta_label_style = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2B6CB0"),
    )
    meta_val_style = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # Title Banner
    story.append(Paragraph(f"{policy['document_id']} — {policy['title']}", title_style))
    story.append(
        Paragraph(
            f"Apex Crest Bank | {policy['department']} | Version {policy['version']} | Effective Date: {policy['effective_date']}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Metadata Box
    meta_data = [
        [
            Paragraph("Document ID:", meta_label_style),
            Paragraph(policy["document_id"], meta_val_style),
            Paragraph("Owner:", meta_label_style),
            Paragraph(policy["owner"], meta_val_style),
        ],
        [
            Paragraph("Classification:", meta_label_style),
            Paragraph(policy["classification"], meta_val_style),
            Paragraph("Effective Date:", meta_label_style),
            Paragraph(policy["effective_date"], meta_val_style),
        ],
        [
            Paragraph("Allowed Roles:", meta_label_style),
            Paragraph(", ".join(policy["allowed_roles"]), meta_val_style),
            Paragraph("References:", meta_label_style),
            Paragraph(
                ", ".join(policy["references"]) if policy["references"] else "None",
                meta_val_style,
            ),
        ],
    ]
    t = Table(meta_data, colWidths=[90, 160, 90, 160])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    # Fictional disclaimer notice
    disclaimer_text = (
        "<b>DEMONSTRATION DOCUMENT NOTICE:</b> This policy document is entirely fictional and created "
        "solely for benchmarking, evaluation, and educational demonstration purposes at Apex Crest Bank. "
        "All figures, limits, and institutional procedures are synthetic and do not constitute actual regulatory requirements."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 6))

    # Numbered Policy Sections
    for s in policy["sections"]:
        heading_text = f"Section {s['section_number']}: {s['heading']}"
        story.append(Paragraph(heading_text, h2_style))
        story.append(Paragraph(s["content"], body_style))
        story.append(Spacer(1, 4))

    doc.build(story, canvasmaker=NumberedCanvas)


def main() -> None:
    """Generate all 20 PDFs and update document_registry.json."""
    policy_sources_path = paths.metadata_dir / "policy_sources.json"
    if not policy_sources_path.exists():
        raise FileNotFoundError(f"Missing source file: {policy_sources_path}")

    with open(policy_sources_path, encoding="utf-8") as f:
        policies = json.load(f)

    if len(policies) != 20:
        raise ValueError(f"Expected exactly 20 policies, found {len(policies)}")

    paths.documents_dir.mkdir(parents=True, exist_ok=True)
    registry_records = []

    for policy in policies:
        doc_id = policy["document_id"]
        filename = f"{doc_id.lower().replace('-', '_')}.pdf"
        out_pdf = paths.documents_dir / filename

        print(f"Generating PDF for {doc_id} -> {filename}...")
        generate_pdf(policy, out_pdf)

        content_hash = compute_source_hash(policy)
        relative_path = f"data/documents/{filename}"

        registry_records.append(
            {
                "document_id": doc_id,
                "title": policy["title"],
                "department": policy["department"],
                "classification": policy["classification"],
                "allowed_roles": policy["allowed_roles"],
                "active_version": policy["version"],
                "effective_date": policy["effective_date"],
                "relative_pdf_path": relative_path,
                "source_content_hash": content_hash,
                "active_status": True,
            }
        )

    registry_path = paths.metadata_dir / "document_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_records, f, indent=2)

    print(f"Generated {len(policies)} PDFs in {paths.documents_dir}")
    print(f"Updated document registry at {registry_path}")


if __name__ == "__main__":
    main()
