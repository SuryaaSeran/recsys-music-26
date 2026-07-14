"""Render paper_wikis/paper_draft.md into the ACM acmart Word template.

Loads the interim ACM template (for its named styles + page setup), clears the
sample body, and rebuilds the paper using acmart styles (Title_document, AbsHead,
Abstract, CCSHead, KeyWordHead, Head1/2/3, Para, AckHead, ReferenceHead,
Bib_entry, TableCaption, Table Grid). Single-column draft mode, matching the
template; the ACM macros produce the final two-column layout.
"""
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path.home() / ".claude/uploads/95ce8712-e6ed-46cb-814b-7ef94e493b61/56402faa-interimlayout.docx"
SRC = ROOT / "paper_wikis/paper_draft.md"
OUT = ROOT / "paper_wikis/paper.docx"


def has_style(doc, name):
    try:
        _ = doc.styles[name]
        return True
    except KeyError:
        return False


def add_runs(paragraph, text):
    """Parse inline **bold**, *italic*, `mono` and add formatted runs."""
    token = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")
    for part in token.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2]); r.bold = True
        elif part.startswith("*") and part.endswith("*"):
            r = paragraph.add_run(part[1:-1]); r.italic = True
        elif part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1]); r.font.name = "Courier New"
        else:
            paragraph.add_run(part)


def main():
    doc = Document(str(TEMPLATE))

    # style resolution with fallbacks
    S = {
        "title": "Title_document" if has_style(doc, "Title_document") else "Title",
        "authors": "Authors",
        "affil": "Affiliation" if has_style(doc, "Affiliation") else "Authors",
        "abshead": "AbsHead", "abstract": "Abstract",
        "ccshead": "CCSHead", "ccs": "CCSDescription",
        "kwhead": "KeyWordHead", "kw": "KeyWords",
        "h1": "Head1", "h2": "Head2", "h3": "Head3",
        "para": "Para", "list": "List Paragraph",
        "quote": "Quote" if has_style(doc, "Quote") else "Para",
        "code": "programCode_display" if has_style(doc, "programCode_display") else "Para",
        "ackhead": "AckHead", "ackpara": "AckPara",
        "refhead": "ReferenceHead", "bib": "Bib_entry" if has_style(doc, "Bib_entry") else "Para",
        "tcap": "TableCaption" if has_style(doc, "TableCaption") else "Caption",
    }

    # clear sample body paragraphs and tables
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    def add(style, text=""):
        p = doc.add_paragraph(style=S[style] if style in S else style)
        if text:
            add_runs(p, text)
        return p

    def add_table(rows):
        header, body = rows[0], rows[1:]
        tbl = doc.add_table(rows=1, cols=len(header))
        tbl.style = "Table Grid"
        for j, cell in enumerate(header):
            c = tbl.rows[0].cells[j]
            c.paragraphs[0].text = ""
            r = c.paragraphs[0].add_run(cell); r.bold = True
            r.font.size = Pt(9)
        for row in body:
            cells = tbl.add_row().cells
            for j, cell in enumerate(row):
                if j >= len(cells):
                    break
                cells[j].paragraphs[0].text = ""
                add_runs(cells[j].paragraphs[0], cell)
                for run in cells[j].paragraphs[0].runs:
                    run.font.size = Pt(9)

    lines = SRC.read_text().splitlines()
    i = 0
    seen_title = False
    section = "front"  # front|abstract|ccs|kw|body|ack|ref
    para_buf = []

    def flush_para():
        nonlocal para_buf
        if not para_buf:
            return
        text = " ".join(para_buf).strip()
        para_buf = []
        if not text:
            return
        if section == "abstract":
            add("abstract", text)
        elif section == "ccs":
            add("ccs", text)
        elif section == "kw":
            add("kw", text)
        elif section == "ack":
            add("ackpara", text)
        elif section == "ref":
            add("bib", text)
        else:
            add("para", text)

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # skip the instructional italic note (may span multiple lines)
        if stripped.startswith("*("):
            while i < len(lines) and not lines[i].strip().endswith(")*"):
                i += 1
            i += 1  # consume the closing line
            continue
        if stripped == "---":
            flush_para(); i += 1; continue

        # table caption: **Table N: ...**  -> TableCaption, placed before its table
        cm = re.match(r"^\*\*(Table \d+:.*)\*\*$", stripped)
        if cm:
            flush_para(); add("tcap", cm.group(1)); i += 1; continue

        # fenced code block
        if stripped.startswith("```"):
            flush_para()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                p = doc.add_paragraph(style=S["code"])
                r = p.add_run(lines[i]); r.font.name = "Courier New"; r.font.size = Pt(9)
                i += 1
            i += 1
            continue

        # table block
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", "".join(cells)):
                    rows.append(cells)
                i += 1
            add_table(rows)
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para()
            level, htext = len(m.group(1)), m.group(2).strip()
            if level == 1 and not seen_title:
                add("title", htext); seen_title = True
                section = "front"
            elif htext.upper() == "ABSTRACT":
                add("abshead", "ABSTRACT"); section = "abstract"
            elif htext.upper().startswith("CCS"):
                add("ccshead", "CCS CONCEPTS"); section = "ccs"
            elif htext.upper().startswith("KEYWORD"):
                add("kwhead", "KEYWORDS"); section = "kw"
            elif htext.upper().startswith("ACKNOWLE"):
                add("ackhead", "ACKNOWLEDGMENTS"); section = "ack"
            elif htext.upper().startswith("REFERENCE"):
                add("refhead", "REFERENCES"); section = "ref"
            else:
                section = "body"
                add({2: "h1", 3: "h2", 4: "h3"}.get(level, "h1"), htext)
            i += 1
            continue

        # front-matter author lines
        if section == "front" and stripped.startswith("**Authors:**"):
            flush_para()
            add("authors", stripped.replace("**Authors:**", "").strip())
            i += 1; continue
        if section == "front" and stripped.startswith("Department Name"):
            flush_para()
            add("affil", stripped)
            i += 1; continue

        # blockquote (render literally; the composite formula contains '*')
        if stripped.startswith(">"):
            flush_para()
            p = doc.add_paragraph(style=S["quote"])
            p.add_run(stripped.lstrip("> ").strip())
            i += 1; continue

        # list items (gather indented continuation lines into one item)
        lm = re.match(r"^(\d+\.|[-*])\s+(.*)$", stripped)
        if lm:
            flush_para()
            item = lm.group(2)
            j = i + 1
            while (j < len(lines) and lines[j].strip()
                   and lines[j][:1] == " "
                   and not re.match(r"^\s*(\d+\.|[-*])\s+", lines[j])
                   and lines[j].strip()[0] not in "#|`"):
                item += " " + lines[j].strip()
                j += 1
            style = "ccs" if section == "ccs" else ("kw" if section == "kw" else "list")
            p = doc.add_paragraph(style=S[style])
            add_runs(p, item)
            i = j
            continue

        # blank line ends a paragraph
        if not stripped:
            flush_para(); i += 1; continue

        para_buf.append(stripped)
        i += 1

    flush_para()
    doc.save(str(OUT))
    print("wrote", OUT)
    # sanity
    d2 = Document(str(OUT))
    print("paragraphs:", len(d2.paragraphs), "tables:", len(d2.tables))


if __name__ == "__main__":
    main()
