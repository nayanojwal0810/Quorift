"""PDF document ingestion and section-based chunking for Quorift."""

import json
from pathlib import Path
import re
from typing import Any

import pymupdf

from quorift.config import paths


def extract_sections_from_pdf(
    pdf_path: Path,
    doc_id: str | None = None,
    expected_sections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract policy sections and page references from a policy PDF using PyMuPDF.

    Strips running headers, footers, metadata-table blocks, and demonstration
    notices from the searchable body, while preserving reading order and text.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Policy PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    page_blocks: list[tuple[int, float, str]] = []

    for page_idx, page in enumerate(doc):
        raw_blocks = page.get_text("blocks")
        # Keep text blocks (type 0) and sort in reading order (vertical y0, then horizontal x0)
        text_blocks = [b for b in raw_blocks if b[6] == 0]
        text_blocks.sort(key=lambda b: (b[1], b[0]))
        for b in text_blocks:
            page_blocks.append((page_idx + 1, b[1], b[4].strip()))

    content_blocks: list[tuple[int, str]] = []
    for pg, y0, txt in page_blocks:
        # Filter footers (bottom margin or disclaimer text)
        if y0 > 740 or "CONFIDENTIAL & PROPRIETARY" in txt:
            continue
        if re.match(r"^Page \d+ of \d+", txt):
            continue

        # Filter running headers (top margin on subsequent pages)
        if pg > 1 and (y0 < 50 or "Apex Crest Bank — Internal Policy" in txt):
            continue

        # Filter first-page document title banner
        if doc_id and re.match(rf"^{re.escape(doc_id)}\s*—", txt):
            continue
        if txt.startswith("Apex Crest Bank |") and "Effective Date:" in txt:
            continue

        # Filter first-page metadata table cells
        if any(
            marker in txt
            for marker in ("Document ID:", "Classification:", "Allowed Roles:", "Owner:", "References:")
        ):
            continue

        # Filter demonstration disclaimer notice
        if "DEMONSTRATION DOCUMENT NOTICE:" in txt:
            continue

        if txt:
            content_blocks.append((pg, txt))

    # Parse numbered section headings and their respective body content
    extracted_sections: list[dict[str, Any]] = []
    current_sec: dict[str, Any] | None = None

    for pg, txt in content_blocks:
        match = re.match(r"^Section\s+(\d+\.\d+):\s*(.*)", txt, re.DOTALL)
        if match:
            if current_sec is not None:
                content_str = " ".join(" ".join(current_sec["content_parts"]).split())
                extracted_sections.append(
                    {
                        "section_number": current_sec["section_number"],
                        "heading": current_sec["heading"],
                        "page_start": current_sec["page_start"],
                        "page_end": current_sec["page_end"],
                        "content": content_str,
                    }
                )
            sec_num = match.group(1).strip()
            sec_heading = " ".join(match.group(2).split())
            current_sec = {
                "section_number": sec_num,
                "heading": sec_heading,
                "page_start": pg,
                "page_end": pg,
                "content_parts": [],
            }
        elif current_sec is not None:
            current_sec["content_parts"].append(txt)
            current_sec["page_end"] = pg

    if current_sec is not None:
        content_str = " ".join(" ".join(current_sec["content_parts"]).split())
        extracted_sections.append(
            {
                "section_number": current_sec["section_number"],
                "heading": current_sec["heading"],
                "page_start": current_sec["page_start"],
                "page_end": current_sec["page_end"],
                "content": content_str,
            }
        )

    # Validate extracted sections against expected metadata if supplied
    if expected_sections is not None:
        expected_nums = [s["section_number"] for s in expected_sections]
        actual_nums = [s["section_number"] for s in extracted_sections]
        if actual_nums != expected_nums:
            raise ValueError(
                f"{doc_id or pdf_path.name}: Section number mismatch. "
                f"Expected {expected_nums}, extracted {actual_nums}"
            )

    for s in extracted_sections:
        if not s["content"].strip():
            raise ValueError(
                f"{doc_id or pdf_path.name} Section {s['section_number']} has empty extracted text"
            )
        if not (1 <= s["page_start"] <= s["page_end"] <= total_pages):
            raise ValueError(
                f"{doc_id or pdf_path.name} Section {s['section_number']} page range invalid: "
                f"{s['page_start']}..{s['page_end']} (total pages {total_pages})"
            )

    return extracted_sections


def build_chunk_text(
    document_id: str,
    title: str,
    section_number: str,
    section_heading: str,
    content: str,
) -> str:
    """Format human-readable chunk text including document and section headers."""
    return f"{document_id} | {title}\nSection {section_number}: {section_heading}\n\n{content}"


def build_chunks(
    registry_path: Path | None = None,
    sources_path: Path | None = None,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Ingest all active policy PDFs and generate structured section chunks with metadata."""
    reg_file = registry_path or (paths.metadata_dir / "document_registry.json")
    src_file = sources_path or (paths.metadata_dir / "policy_sources.json")
    root = project_root or paths.project_root

    if not reg_file.exists():
        raise FileNotFoundError(f"Document registry not found: {reg_file}")
    if not src_file.exists():
        raise FileNotFoundError(f"Policy sources not found: {src_file}")

    with open(reg_file, encoding="utf-8") as f:
        registry = json.load(f)
    with open(src_file, encoding="utf-8") as f:
        policy_sources = {p["document_id"]: p for p in json.load(f)}

    active_registry = [r for r in registry if r.get("active_status", True)]
    if len(active_registry) != 20:
        raise ValueError(f"Expected exactly 20 active documents, found {len(active_registry)}")

    chunks: list[dict[str, Any]] = []

    for reg_entry in active_registry:
        doc_id = reg_entry["document_id"]
        source_meta = policy_sources.get(doc_id)
        if not source_meta:
            raise ValueError(f"Document {doc_id} in registry missing from policy sources")

        rel_pdf = reg_entry["relative_pdf_path"].replace("\\", "/")
        full_pdf_path = root / rel_pdf
        if not full_pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found for {doc_id}: {full_pdf_path}")

        extracted_sections = extract_sections_from_pdf(
            pdf_path=full_pdf_path,
            doc_id=doc_id,
            expected_sections=source_meta.get("sections"),
        )

        version = reg_entry.get("active_version", source_meta.get("version", "1.0"))
        title = reg_entry.get("title", source_meta.get("title"))
        department = reg_entry.get("department", source_meta.get("department"))
        allowed_roles = reg_entry.get("allowed_roles", source_meta.get("allowed_roles"))
        classification = source_meta.get("classification", "Internal - Confidential")
        effective_date = source_meta.get("effective_date", "2026-01-15")
        source_hash = reg_entry.get("source_content_hash", "")

        for s in extracted_sections:
            sec_num = s["section_number"]
            sec_heading = s["heading"]
            content = s["content"]
            chunk_id = f"{doc_id}|{version}|{sec_num}"
            chunk_text = build_chunk_text(
                document_id=doc_id,
                title=title,
                section_number=sec_num,
                section_heading=sec_heading,
                content=content,
            )

            chunk = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": title,
                "department": department,
                "section_number": sec_num,
                "section_heading": sec_heading,
                "version": version,
                "effective_date": effective_date,
                "page_start": s["page_start"],
                "page_end": s["page_end"],
                "allowed_roles": allowed_roles,
                "classification": classification,
                "source_pdf": rel_pdf,
                "source_hash": source_hash,
                "text": chunk_text,
            }
            chunks.append(chunk)

    # Deterministic sorting by document ID and numerical section order
    def sort_key(c: dict[str, Any]) -> tuple[str, float]:
        sec_val = 0.0
        try:
            sec_val = float(c["section_number"])
        except ValueError:
            pass
        return (c["document_id"], sec_val)

    chunks.sort(key=sort_key)

    # Invariant validations
    chunk_ids = [c["chunk_id"] for c in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected in generated chunks")
    if len(chunks) != 120:
        raise ValueError(f"Expected 120 chunks across 20 documents, generated {len(chunks)}")

    return chunks


def save_chunks_jsonl(chunks: list[dict[str, Any]], output_path: Path) -> None:
    """Write chunk objects to a JSONL file in stable order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_chunks_jsonl(input_path: Path) -> list[dict[str, Any]]:
    """Read chunk objects from a JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {input_path}")
    chunks: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                chunks.append(json.loads(line_str))
    return chunks
