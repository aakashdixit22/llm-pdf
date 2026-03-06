"""
ETL Pipeline: PDF Processor
Extracts text, tables, and metadata from the Cyber Ireland 2022 Report.
Uses pdfplumber for text+tables and PyMuPDF for supplementary metadata.
"""

import os
import json
import re
import hashlib
from typing import Optional
from dataclasses import dataclass, field, asdict

import pdfplumber
import fitz  # PyMuPDF


@dataclass
class TextChunk:
    """A chunk of text extracted from the PDF."""
    chunk_id: str
    page_number: int
    content: str
    chunk_type: str  # "text", "table", "table_summary"
    section: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class ExtractedTable:
    """A table extracted from the PDF."""
    table_id: str
    page_number: int
    raw_data: list  # list of lists (rows x cols)
    caption: str = ""
    markdown: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _generate_chunk_id(page: int, index: int, chunk_type: str) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"p{page}_{chunk_type}_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _clean_text(text: str) -> str:
    """Clean extracted text: remove duplicate characters from OCR artifacts."""
    # Fix common OCR double-character artifacts like "CCYYBBEERR" -> "CYBER"
    # These appear on certain pages of the PDF
    cleaned = text
    # Remove page header/footer patterns
    cleaned = re.sub(
        r'\d+\s+CYBER IRELAND 2022 EDITION\s+STATE OF THE CYBER SECURITY SECTOR IN IRELAND\s+\d+',
        '', cleaned
    )
    # Remove double-char OCR artifacts (e.g., "CCYYBBEERR IIRREELLAANNDD")
    cleaned = re.sub(
        r'([A-Z])\1([A-Z])\2([A-Z])\3([A-Z])\4([A-Z])\5',
        lambda m: m.group(1) + m.group(2) + m.group(3) + m.group(4) + m.group(5),
        cleaned
    )
    # Normalize whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    return cleaned.strip()


def _table_to_markdown(table: list) -> str:
    """Convert a raw table (list of lists) to Markdown format."""
    if not table or len(table) < 2:
        return ""

    # Clean None values
    clean_table = []
    for row in table:
        clean_row = [str(cell).strip() if cell else "" for cell in row]
        # Skip rows that are all empty
        if any(cell for cell in clean_row):
            clean_table.append(clean_row)

    if not clean_table:
        return ""

    # Normalize column count
    max_cols = max(len(row) for row in clean_table)
    for row in clean_table:
        while len(row) < max_cols:
            row.append("")

    # Build markdown
    lines = []
    # Header
    lines.append("| " + " | ".join(clean_table[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    # Data rows
    for row in clean_table[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _table_to_text_summary(table: list, caption: str, page: int) -> str:
    """Generate a text summary of a table for semantic search."""
    if not table:
        return ""

    # Clean None values
    clean_table = []
    for row in table:
        clean_row = [str(cell).strip() if cell else "" for cell in row]
        if any(cell for cell in clean_row):
            clean_table.append(clean_row)

    if not clean_table:
        return ""

    summary_parts = []
    if caption:
        summary_parts.append(f"Table: {caption} (Page {page})")

    # Try to create a natural language summary
    header = clean_table[0] if clean_table else []
    for row in clean_table[1:]:
        row_desc = []
        for i, cell in enumerate(row):
            if cell and i < len(header) and header[i]:
                row_desc.append(f"{header[i]}: {cell}")
            elif cell:
                row_desc.append(cell)
        if row_desc:
            summary_parts.append("; ".join(row_desc))

    return "\n".join(summary_parts)


def _detect_section(text: str) -> str:
    """Detect section heading from text content."""
    # Look for numbered section patterns like "3.4 LOCATION OF..."
    match = re.search(r'(\d+\.\d+)\s+([A-Z][A-Z\s&,]+)', text[:500])
    if match:
        return f"{match.group(1)} {match.group(2).strip()}"

    # Look for all-caps headings
    match = re.search(r'^([A-Z][A-Z\s&,]{10,})', text[:200], re.MULTILINE)
    if match:
        return match.group(1).strip()[:80]

    return ""


def extract_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> tuple[list[TextChunk], list[ExtractedTable]]:
    """
    Extract text and tables from the PDF.

    Returns:
        tuple: (text_chunks, extracted_tables)
    """
    chunks: list[TextChunk] = []
    tables: list[ExtractedTable] = []

    print(f"[ETL] Opening PDF: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[ETL] Total pages: {total_pages}")

        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            text = page.extract_text() or ""
            page_tables = page.extract_tables() or []

            # Clean text
            cleaned_text = _clean_text(text)
            section = _detect_section(cleaned_text)

            # --- Extract tables as atomic units ---
            for t_idx, raw_table in enumerate(page_tables):
                # Skip empty or tiny tables
                clean_rows = [
                    row for row in raw_table
                    if row and any(cell and str(cell).strip() for cell in row)
                ]
                if len(clean_rows) < 2:
                    continue

                table_id = _generate_chunk_id(page_num, t_idx, "table")
                caption = _find_table_caption(cleaned_text, t_idx)
                markdown = _table_to_markdown(clean_rows)
                text_summary = _table_to_text_summary(clean_rows, caption, page_num)

                table_obj = ExtractedTable(
                    table_id=table_id,
                    page_number=page_num,
                    raw_data=clean_rows,
                    caption=caption,
                    markdown=markdown,
                    metadata={"section": section, "num_rows": len(clean_rows)},
                )
                tables.append(table_obj)

                # Create a chunk for the table summary (for vector search)
                if text_summary:
                    table_chunk = TextChunk(
                        chunk_id=f"ts_{table_id}",
                        page_number=page_num,
                        content=text_summary,
                        chunk_type="table_summary",
                        section=section,
                        metadata={
                            "table_id": table_id,
                            "caption": caption,
                            "markdown": markdown,
                        },
                    )
                    chunks.append(table_chunk)

                # Also create a chunk for the raw markdown table
                if markdown:
                    md_chunk = TextChunk(
                        chunk_id=f"tm_{table_id}",
                        page_number=page_num,
                        content=f"[TABLE on Page {page_num}] {caption}\n\n{markdown}",
                        chunk_type="table",
                        section=section,
                        metadata={
                            "table_id": table_id,
                            "caption": caption,
                            "raw_data_json": json.dumps(clean_rows),
                        },
                    )
                    chunks.append(md_chunk)

            # --- Chunk the text content ---
            if cleaned_text:
                text_chunks = _split_text(
                    cleaned_text, page_num, section, chunk_size, chunk_overlap
                )
                chunks.extend(text_chunks)

            if page_num % 5 == 0:
                print(f"[ETL] Processed page {page_num}/{total_pages}")

    print(
        f"[ETL] Extraction complete: {len(chunks)} chunks, {len(tables)} tables"
    )
    return chunks, tables


def _find_table_caption(text: str, table_idx: int) -> str:
    """Try to find a table caption in the text."""
    # Look for patterns like "TABLE 3.1 ..." or "FIGURE 3.1 ..."
    patterns = [
        r'(TABLE\s+\d+[\.\d]*\s+[A-Z][A-Z\s&,\(\)]+)',
        r'(FIGURE\s+\d+[\.\d]*\s+[A-Z][A-Z\s&,\(\)]+)',
    ]
    captions = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        captions.extend(matches)

    if table_idx < len(captions):
        return captions[table_idx].strip()[:200]
    elif captions:
        return captions[0].strip()[:200]
    return ""


def _split_text(
    text: str,
    page_num: int,
    section: str,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    """Split text into overlapping chunks while trying to respect sentence boundaries."""
    if not text or len(text) < 50:
        return []

    chunks = []
    # Split by paragraphs first
    paragraphs = re.split(r'\n\n+', text)
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            # Save current chunk
            if current_chunk and len(current_chunk) > 50:
                chunk_id = _generate_chunk_id(page_num, chunk_idx, "text")
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    page_number=page_num,
                    content=current_chunk,
                    chunk_type="text",
                    section=section,
                    metadata={"char_count": len(current_chunk)},
                ))
                chunk_idx += 1

                # Start new chunk with overlap from end of previous
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

    # Don't forget the last chunk
    if current_chunk and len(current_chunk) > 50:
        chunk_id = _generate_chunk_id(page_num, chunk_idx, "text")
        chunks.append(TextChunk(
            chunk_id=chunk_id,
            page_number=page_num,
            content=current_chunk,
            chunk_type="text",
            section=section,
            metadata={"char_count": len(current_chunk)},
        ))

    return chunks


def save_extraction_results(
    chunks: list[TextChunk],
    tables: list[ExtractedTable],
    output_dir: str,
):
    """Save extraction results as JSON files for inspection."""
    os.makedirs(output_dir, exist_ok=True)

    chunks_data = [c.to_dict() for c in chunks]
    tables_data = [t.to_dict() for t in tables]

    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables_data, f, indent=2, ensure_ascii=False)

    print(f"[ETL] Saved {len(chunks_data)} chunks and {len(tables_data)} tables to {output_dir}")


if __name__ == "__main__":
    pdf_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf",
    )
    chunks, tables = extract_pdf(pdf_path)
    save_extraction_results(
        chunks, tables,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
    )
