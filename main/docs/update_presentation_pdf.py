"""Refresh the final submission slide in the exported project PDF.

PowerPoint/LibreOffice is not available in every reproducibility environment, so this
small utility replaces only the final slide with a vector PDF page that matches the
presentation's research-only visual language. The source PPTX remains authoritative for
future full exports.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "GlycoTwin_CGMacros_Prototype.pdf"
TEMP_PDF = ROOT / ".GlycoTwin_CGMacros_Prototype.final-slide.pdf"

PAGE_W = 960
PAGE_H = 540
BACKGROUND = HexColor("#5A411C")
PANEL = HexColor("#F5F6E8")
INK = HexColor("#4C3515")
WHITE = HexColor("#FFFFFF")


def paragraph(c: canvas.Canvas, text: str, x: float, y: float, w: float, h: float, size: float, color=INK, leading=None, bold=False):
    style = ParagraphStyle(
        name="slide",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=leading or size * 1.25,
        textColor=color,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    para = Paragraph(text, style)
    para.wrapOn(c, w, h)
    para.drawOn(c, x, y + h - para.height)


def panel(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, body: list[str]) -> None:
    c.setFillColor(PANEL)
    c.roundRect(x, y, w, h, 2, fill=1, stroke=0)
    paragraph(c, title, x + 16, y + h - 42, w - 32, 24, 16, color=INK, bold=True)
    body_text = "<br/><br/>".join(body)
    paragraph(c, body_text, x + 16, y + 18, w - 32, h - 62, 13, color=INK, leading=18)


def make_page() -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(PAGE_W, PAGE_H))
    c.setFillColor(BACKGROUND)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    paragraph(c, "GLYCOTWIN  /  RESEARCH PROTOTYPE", 48, 490, 860, 24, 14, color=WHITE, bold=True)
    paragraph(c, "Conclusion and submission details", 48, 424, 860, 48, 32, color=WHITE, bold=True)
    paragraph(
        c,
        "GlycoTwin demonstrates a reproducible multimodal forecasting workflow; the evidence supports further research, not clinical use.",
        48,
        382,
        860,
        28,
        17,
        color=WHITE,
    )

    panel(
        c,
        40,
        55,
        285,
        300,
        "What the prototype shows",
        [
            "Static baseline context plus CGM, heart rate, activity, meals, and time-of-day can be fused at forecast timestamps.",
            "Grouped evaluation, same-fold baselines, chronological personalization, calibration, and episode workload expose trade-offs.",
            "The architecture matches the pipeline: align → fuse 41 dynamic + 12 baseline features (53 total) → HGB → forecast → dashboard/evaluation.",
        ],
    )
    panel(
        c,
        355,
        55,
        285,
        300,
        "Team",
        [
            "Team name: Sage",
            "Team lead and member: Manda Rohit Vardhan — Project Lead and Developer",
            "College: Amrita Vishwa Vidyapeetham, Hyderabad",
            "Incubator: None",
        ],
    )
    panel(
        c,
        670,
        55,
        250,
        285,
        "Submission",
        [
            "Demo video: unlisted YouTube link to be added (15–20 minutes)",
            "Repository: github.com/Rohit006-sage/sage-glycotwin-cgmacros",
            "Dataset citation: CGMacros v1.0.0, PhysioNet, DOI 10.13026/3z8q-x658.",
            "Retain the research-only notice and dataset/privacy review before sharing.",
        ],
    )
    paragraph(c, "CGMacros retrospective PoC  •  not for clinical use  •  no diagnosis, treatment, triage, or live monitoring", 48, 28, 860, 18, 11, color=WHITE)
    c.showPage()
    c.save()
    return buffer.getvalue()


def main() -> None:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(PDF_PATH)
    updated_page = fitz.open(stream=make_page(), filetype="pdf")
    source = fitz.open(PDF_PATH)
    if len(source) == 0:
        raise ValueError("Project PDF has no pages")
    source.delete_page(len(source) - 1)
    source.insert_pdf(updated_page)
    source.save(TEMP_PDF)
    source.close()
    updated_page.close()
    TEMP_PDF.replace(PDF_PATH)
    print(f"Updated final submission page in {PDF_PATH}")


if __name__ == "__main__":
    main()
