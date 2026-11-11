#!/usr/bin/env python3
"""Export the bold-marked revised manuscript from Markdown to editable DOCX."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


BOLD = "**"


def set_run_font(run, *, size: float = 11.0, name: str = "Times New Roman") -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def add_inline(paragraph, text: str, *, force_bold: bool = False, force_italic: bool = False) -> None:
    """Render the small Markdown subset used by the manuscript."""
    cursor = 0
    plain = []

    def flush() -> None:
        nonlocal plain
        if plain:
            run = paragraph.add_run("".join(plain))
            if force_bold:
                run.bold = True
            if force_italic:
                run.italic = True
            set_run_font(run)
            plain = []

    while cursor < len(text):
        if text.startswith(BOLD, cursor):
            end = text.find(BOLD, cursor + 2)
            if end >= 0:
                flush()
                add_inline(
                    paragraph,
                    text[cursor + 2 : end],
                    force_bold=True,
                    force_italic=force_italic,
                )
                cursor = end + 2
                continue
        if text[cursor] == "`":
            end = text.find("`", cursor + 1)
            if end >= 0:
                flush()
                run = paragraph.add_run(text[cursor + 1 : end])
                if force_bold:
                    run.bold = True
                if force_italic:
                    run.italic = True
                set_run_font(run, name="Courier New", size=10.0)
                cursor = end + 1
                continue
        if text[cursor] == "^":
            end = text.find("^", cursor + 1)
            if end >= 0:
                flush()
                run = paragraph.add_run(text[cursor + 1 : end])
                if force_bold:
                    run.bold = True
                if force_italic:
                    run.italic = True
                run.font.superscript = True
                set_run_font(run, size=9.0)
                cursor = end + 1
                continue
        if text[cursor] == "*" and not text.startswith(BOLD, cursor):
            end = text.find("*", cursor + 1)
            if end >= 0:
                flush()
                add_inline(
                    paragraph,
                    text[cursor + 1 : end],
                    force_bold=force_bold,
                    force_italic=True,
                )
                cursor = end + 1
                continue
        plain.append(text[cursor])
        cursor += 1
    flush()


def is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?[\s:|-]+\|?", line))


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    table_header = OxmlElement("w:tblHeader")
    table_header.set(qn("w:val"), "true")
    tr_pr.append(table_header)


def add_table(document: Document, lines: list[str]) -> None:
    rows = [table_cells(line) for line in lines if not is_table_separator(line)]
    if not rows:
        return
    column_count = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    table.autofit = True
    repeat_table_header(table.rows[0])
    for row_index, values in enumerate(rows):
        for column_index in range(column_count):
            cell = table.cell(row_index, column_index)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if column_index in {0, 1}
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            add_inline(paragraph, values[column_index] if column_index < len(values) else "")
            for run in paragraph.runs:
                set_run_font(run, size=8.5)
            if row_index == 0:
                shade_cell(cell, "D9EAF7")
    document.add_paragraph()


def add_page_number(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.extend([field_begin, instruction, field_end])
    set_run_font(run, size=9.0)


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    add_page_number(section)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)

    for level in range(1, 4):
        style = document.styles[f"Heading {level}"]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
        style.font.size = Pt({1: 14, 2: 12, 3: 11}[level])
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(4)


def add_figure(document: Document, block: str, source_dir: Path) -> None:
    lines = block.splitlines()
    match = re.match(r"^!\[(.*)\]\((.*)\)$", lines[0])
    if not match:
        return
    alt_text, relative_path = match.groups()
    image_path = source_dir / relative_path
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(image_path), width=Inches(6.4))
    caption = lines[1] if len(lines) > 1 else f"**{alt_text}**"
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_inline(caption_paragraph, caption)
    for run in caption_paragraph.runs:
        run.italic = True
        set_run_font(run, size=9.5)


def export(source: Path, output: Path) -> None:
    document = Document()
    configure_document(document)
    document.core_properties.title = "Revised manuscript with changes in bold"
    document.core_properties.subject = "Audited multilingual abusive-language study"

    blocks = source.read_text(encoding="utf-8").split("\n\n")
    for block_index, raw_block in enumerate(blocks):
        block = raw_block.strip()
        if not block:
            continue
        lines = block.splitlines()
        first = lines[0]

        if first.startswith("!["):
            add_figure(document, block, source.parent)
            continue
        if first.startswith("|"):
            add_table(document, lines)
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", first)
        if heading:
            markdown_level = len(heading.group(1))
            if markdown_level == 1:
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_after = Pt(12)
                add_inline(paragraph, heading.group(2))
                for run in paragraph.runs:
                    run.bold = True
                    set_run_font(run, size=15.0)
            else:
                paragraph = document.add_heading(level=min(markdown_level - 1, 3))
                add_inline(paragraph, heading.group(2))
            continue
        if all(re.match(r"^[-*]\s+", line) for line in lines):
            for line in lines:
                paragraph = document.add_paragraph(style="List Bullet")
                add_inline(paragraph, re.sub(r"^[-*]\s+", "", line))
            continue
        if all(re.match(r"^\d+\.\s+", line) for line in lines):
            for line in lines:
                paragraph = document.add_paragraph(style="List Number")
                add_inline(paragraph, re.sub(r"^\d+\.\s+", "", line))
            continue
        if first.startswith("> "):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.right_indent = Inches(0.25)
            paragraph.paragraph_format.space_after = Pt(8)
            add_inline(paragraph, " ".join(line.removeprefix("> ") for line in lines))
            continue

        paragraph = document.add_paragraph()
        if block_index in {2, 3}:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        text = " ".join(line.strip() for line in lines)
        if text.startswith("*Corresponding author:"):
            text = text[1:]
            add_inline(paragraph, text, force_italic=True)
        else:
            add_inline(paragraph, text)

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("manuscript/REVISED_MANUSCRIPT_CHANGES_BOLDED.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/REVISED_MANUSCRIPT_CHANGES_BOLDED.docx"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(args.source, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
