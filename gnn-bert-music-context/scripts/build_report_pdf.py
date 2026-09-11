"""Build printable HTML (and PDF if reportlab available) from final_report.md."""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "report" / "final_report.md"
HTML_OUT = ROOT / "report" / "final_report.html"
PDF_OUT = ROOT / "report" / "final_report.pdf"


def md_to_html(t: str) -> str:
    lines = []
    in_code = False
    in_table = False
    for line in t.splitlines():
        if line.startswith("```"):
            if not in_code:
                lines.append("<pre>")
                in_code = True
            else:
                lines.append("</pre>")
                in_code = False
            continue
        if in_code:
            lines.append(html.escape(line))
            continue
        if line.startswith("|") and "|" in line[1:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and set(cells[0]).issubset(set("-: ")):
                continue
            if not in_table:
                lines.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            rows = "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells)
            lines.append("<tr>" + rows + "</tr>")
            continue
        else:
            if in_table:
                lines.append("</table>")
                in_table = False
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.strip() == "---":
            lines.append("<hr>")
        elif line.strip() == "":
            lines.append("")
        else:
            s = html.escape(line)
            s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
            s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
            lines.append(f"<p>{s}</p>")
    if in_table:
        lines.append("</table>")
    if in_code:
        lines.append("</pre>")
    return "\n".join(lines)


def main() -> None:
    text = MD.read_text(encoding="utf-8")
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Final Report — CSE425 Section 02</title>
<style>
body{{font-family:Georgia,'Times New Roman',serif;max-width:820px;margin:2rem auto;line-height:1.5;padding:0 1.2rem;color:#111}}
h1{{font-size:1.6rem}} h2{{font-size:1.25rem;margin-top:1.6rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{border:1px solid #bbb;padding:6px 8px;text-align:left}}
code,pre{{background:#f6f6f6}} pre{{padding:12px;overflow:auto;font-size:.9rem}}
@media print{{body{{max-width:100%;margin:0}}}}
</style></head><body>
{md_to_html(text)}
<p><em>Authors: Ridhwanur Rahman Khan (22301143), Jannatul Bushra Maisha (21301498) — CSE425 Section 02.
Print this page to PDF (Ctrl+P) if LaTeX is unavailable.</em></p>
</body></html>
"""
    HTML_OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {HTML_OUT}")

    # Optional PDF via reportlab (plain text pages)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        styles = getSampleStyleSheet()
        doc_pdf = SimpleDocTemplate(str(PDF_OUT), pagesize=A4, title="CSE425 Final Report")
        story = []
        for line in text.splitlines():
            if line.startswith("# "):
                story.append(Paragraph(html.escape(line[2:]), styles["Title"]))
            elif line.startswith("## "):
                story.append(Paragraph(html.escape(line[3:]), styles["Heading1"]))
            elif line.startswith("### "):
                story.append(Paragraph(html.escape(line[4:]), styles["Heading2"]))
            elif line.strip() == "":
                story.append(Spacer(1, 0.1 * inch))
            else:
                story.append(Paragraph(html.escape(line).replace("**", ""), styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))
        doc_pdf.build(story)
        print(f"Wrote {PDF_OUT}")
    except Exception as e:
        print(f"PDF via reportlab skipped: {e}")
        print("Open final_report.html and Print → Save as PDF, or compile final_report.tex on Overleaf.")


if __name__ == "__main__":
    main()
