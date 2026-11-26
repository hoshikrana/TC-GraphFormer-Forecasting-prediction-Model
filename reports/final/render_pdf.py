"""
Script to render TC_GraphFormer_Research_Report.md into a high-quality academic PDF report.
Uses fpdf2 with Segoe UI and Consolas fonts, supporting unicode math, tables, figures, code blocks.
"""

import os
import re
import sys
from pathlib import Path
from fpdf import FPDF
from fpdf.fonts import FontFace
from fpdf.enums import TableCellFillMode, Align

def sanitize_math(text: str) -> str:
    """Translates KaTeX / LaTeX math strings into clean unicode representation."""
    # Common macro replacements
    replacements = [
        (r'\\mathbf\{([^}]+)\}', r'\1'),
        (r'\\text\{([^}]+)\}', r'\1'),
        (r'\\mathbb\{I\}_\{([^}]+)\}', r'I(\1)'),
        (r'\\mathbb\{I\}', r'I'),
        (r'\\mathcal\{L\}_\{([^}]+)\}', r'L_\1'),
        (r'\\mathcal\{L\}', r'L'),
        (r'\\Delta', 'Δ'),
        (r'\\tau', 'τ'),
        (r'\\theta', 'θ'),
        (r'\\mu', 'μ'),
        (r'\\rho', 'ρ'),
        (r'\\sigma', 'σ'),
        (r'\\ge', '≥'),
        (r'\\le', '≤'),
        (r'\\pm', '±'),
        (r'\\times', '×'),
        (r'\\in', ' in '),
        (r'\\approx', '≈'),
        (r'\\to', '→'),
        (r'\\cdot', '·'),
        (r'\\sum', 'Σ'),
        (r'\\hat\{([^}]+)\}', r'\1^'),
        (r'\\bar\{([^}]+)\}', r'\1_avg'),
        (r'V_\{?\\max\}?', 'V_max'),
        (r'P_\{?\\min\}?', 'P_min'),
        (r'\\left\(', '('),
        (r'\\right\)', ')'),
        (r'\\left\[', '['),
        (r'\\right\]', ']'),
        (r'\\ln', 'ln'),
        (r'\\cos', 'cos'),
        (r'\\sin', 'sin'),
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1 / \2)'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    # Strip remaining single $ or $$
    text = text.replace('$$', '')
    text = text.replace('$', '')
    return text

class AcademicReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=18)
        
        # Load Windows system fonts
        font_dir = "C:/Windows/Fonts"
        self.add_font("Segoe", "", os.path.join(font_dir, "segoeui.ttf"))
        self.add_font("Segoe", "B", os.path.join(font_dir, "segoeuib.ttf"))
        self.add_font("Segoe", "I", os.path.join(font_dir, "segoeuii.ttf"))
        self.add_font("Consolas", "", os.path.join(font_dir, "consola.ttf"))
        
        self.title_rendered = False

    def header(self):
        if self.page_no() > 1:
            self.set_font("Segoe", "I", 8)
            self.set_text_color(100, 116, 139) # slate-500
            self.cell(0, 5, "TC-GraphFormer: Academic Research & Generalization Engineering Report", align="L")
            self.ln(5)
            self.set_draw_color(226, 232, 240)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(226, 232, 240)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Segoe", "", 8)
        self.set_text_color(148, 163, 184) # slate-400
        self.cell(0, 5, "NOAA TC-PRIMED 100 GB Expansion Benchmark — Academic University Submission", align="L")
        self.set_x(-35)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="R")

def render_markdown_to_pdf(md_path: Path, pdf_path: Path):
    pdf = AcademicReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    with open(md_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\r\n") for line in f]

    i = 0
    n = len(lines)
    
    header_font_face = FontFace(family="Segoe", emphasis="B", size_pt=7.5, fill_color=(241, 245, 249))
    cell_font_face = FontFace(family="Segoe", emphasis="", size_pt=7.5)
    bold_cell_font_face = FontFace(family="Segoe", emphasis="B", size_pt=7.5)

    while i < n:
        line = lines[i]
        
        # Blank line
        if not line.strip():
            i += 1
            continue
            
        # Title (# ...)
        if line.startswith("# "):
            title_text = line[2:].strip()
            pdf.set_font("Segoe", "B", 18)
            pdf.set_text_color(15, 23, 42) # slate-900
            pdf.multi_cell(0, 8, title_text, align="L")
            pdf.ln(3)
            i += 1
            continue
            
        # Section (## ...)
        if line.startswith("## "):
            sec_text = line[3:].strip()
            pdf.ln(4)
            # Decorative section header
            pdf.set_font("Segoe", "B", 13)
            pdf.set_text_color(30, 58, 138) # dark blue
            pdf.cell(0, 7, sec_text, align="L")
            pdf.ln(7)
            pdf.set_draw_color(191, 219, 254) # blue-200
            pdf.set_line_width(0.4)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue
            
        # Subsection (### ...)
        if line.startswith("### "):
            subsec_text = line[4:].strip()
            pdf.ln(3)
            pdf.set_font("Segoe", "B", 10.5)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 6, subsec_text, align="L")
            pdf.ln(6)
            i += 1
            continue
            
        # Sub-subsection (#### ...)
        if line.startswith("#### "):
            subsubsec_text = line[5:].strip()
            pdf.ln(2)
            pdf.set_font("Segoe", "B", 9.5)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(0, 5, subsubsec_text, align="L")
            pdf.ln(5)
            i += 1
            continue

        # Horizontal Rule (---)
        if line.strip() == "---":
            pdf.ln(2)
            pdf.set_draw_color(226, 232, 240)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # Image ![alt](figures/...)
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', line.strip())
        if img_match:
            alt_text, img_rel_path = img_match.groups()
            img_full_path = (md_path.parent / img_rel_path).resolve()
            if img_full_path.exists():
                pdf.ln(2)
                # Compute width to fit page neatly (max 165mm)
                img_w = 165
                x_pos = (pdf.w - img_w) / 2
                pdf.image(str(img_full_path), x=x_pos, w=img_w)
                pdf.ln(2)
                # Next line might be caption
                if i + 1 < n and lines[i+1].startswith("*Figure"):
                    caption = lines[i+1].strip().strip("*")
                    # Clean markdown links from caption
                    caption = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', caption)
                    pdf.set_font("Segoe", "I", 8)
                    pdf.set_text_color(100, 116, 139)
                    pdf.multi_cell(0, 4.5, caption, align="C")
                    pdf.ln(3)
                    i += 2
                    continue
            i += 1
            continue

        # Code block (``` ... ```)
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1 # skip ending ```
            code_text = "\n".join(code_lines)
            
            pdf.set_font("Consolas", "", 7.5)
            pdf.set_text_color(30, 41, 59)
            pdf.set_fill_color(248, 250, 252)
            pdf.set_draw_color(226, 232, 240)
            pdf.set_line_width(0.2)
            
            # Print code box
            pdf.multi_cell(0, 3.8, code_text, border=1, fill=True, align="L")
            pdf.ln(3)
            continue

        # Markdown Table (| ... |)
        if line.startswith("|") and line.endswith("|"):
            table_lines = []
            while i < n and lines[i].startswith("|") and lines[i].endswith("|"):
                table_lines.append(lines[i])
                i += 1
                
            if len(table_lines) >= 2:
                # Parse header
                headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
                # Parse separator (check alignments)
                # Parse data rows
                data_rows = []
                for row_line in table_lines[2:]:
                    cells = [c.strip() for c in row_line.strip("|").split("|")]
                    # Match header length
                    if len(cells) < len(headers):
                        cells += [""] * (len(headers) - len(cells))
                    elif len(cells) > len(headers):
                        cells = cells[:len(headers)]
                    data_rows.append(cells)
                    
                # Render table
                pdf.set_font("Segoe", "", 7.5)
                # Calculate column widths
                num_cols = len(headers)
                avail_w = pdf.w - pdf.l_margin - pdf.r_margin
                
                # Estimate widths based on content lengths
                col_lens = [len(h) for h in headers]
                for r in data_rows:
                    for idx, c in enumerate(r):
                        col_lens[idx] = max(col_lens[idx], len(c))
                tot_len = sum(col_lens)
                col_widths = [max(16.0, (l / tot_len) * avail_w) for l in col_lens]
                # Re-normalize to exact available width
                scale = avail_w / sum(col_widths)
                col_widths = [w * scale for w in col_widths]

                with pdf.table(col_widths=col_widths, text_align="LEFT", borders_layout="ALL", cell_fill_mode=TableCellFillMode.ROWS) as tbl:
                    # Header row
                    hdr_row = tbl.row()
                    for idx, h in enumerate(headers):
                        clean_h = sanitize_math(h).replace("**", "")
                        hdr_row.cell(clean_h, style=header_font_face)
                    
                    # Data rows
                    for r_idx, r in enumerate(data_rows):
                        d_row = tbl.row()
                        for c_idx, cell_val in enumerate(r):
                            clean_cell = sanitize_math(cell_val)
                            is_bold = "**" in clean_cell or r_idx == len(data_rows) - 1
                            clean_cell = clean_cell.replace("**", "").replace("*", "")
                            
                            # Alignment: right for numeric/percentage, left for text
                            c_align = "RIGHT" if any(char.isdigit() for char in clean_cell) and not "/" in clean_cell and len(clean_cell) < 20 else "LEFT"
                            if c_idx == 0:
                                c_align = "LEFT"
                                
                            d_row.cell(clean_cell, style=bold_cell_font_face if is_bold else cell_font_face, align=c_align)
                pdf.ln(3)
                continue

        # Block math $$ ... $$
        if line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 4:
            math_text = line.strip()[2:-2].strip()
            clean_m = sanitize_math(math_text)
            pdf.ln(1)
            pdf.set_font("Segoe", "I", 8.5)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 5, clean_m, align="C")
            pdf.ln(5)
            i += 1
            continue

        # Bullet or numbered list
        list_match = re.match(r'^(\d+\.|\*|\-)\s+(.*)$', line)
        if list_match:
            bullet, item_text = list_match.groups()
            clean_item = sanitize_math(item_text)
            # Remove markdown file links but keep text
            clean_item = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_item)
            
            pdf.set_font("Segoe", "", 8.5)
            pdf.set_text_color(51, 65, 85)
            bullet_sym = "•" if bullet in ["*", "-"] else bullet
            
            # Indented list item
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(0, 4.5, f"{bullet_sym}  {clean_item}", markdown=True)
            pdf.ln(1)
            i += 1
            continue

        # Normal paragraph or metadata line
        para_text = sanitize_math(line)
        para_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', para_text)
        
        pdf.set_font("Segoe", "", 8.5)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 4.5, para_text, markdown=True)
        pdf.ln(2)
        i += 1

    # Output PDF
    pdf.output(str(pdf_path))
    print(f"Successfully generated academic research report PDF: {pdf_path}")
    print(f"Page count: {pdf.page_no()}")

if __name__ == "__main__":
    report_md = Path(r"D:\Enjula uchoi\TC-GraphFormer-verified\reports\final\TC_GraphFormer_Research_Report.md")
    report_pdf = Path(r"D:\Enjula uchoi\TC-GraphFormer-verified\reports\final\TC_GraphFormer_Research_Report.pdf")
    render_markdown_to_pdf(report_md, report_pdf)
