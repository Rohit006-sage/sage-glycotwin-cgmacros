from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "architecture.pptx"
MMD = ROOT / "architecture.mmd"

FOREST = RGBColor(10, 59, 44)
GREEN = RGBColor(22, 116, 91)
MINT = RGBColor(234, 243, 238)
PALE = RGBColor(245, 247, 246)
AMBER = RGBColor(210, 138, 46)
AMBER_PALE = RGBColor(255, 245, 229)
INK = RGBColor(25, 52, 43)
MUTED = RGBColor(97, 119, 110)
GRID = RGBColor(203, 217, 208)
WHITE = RGBColor(255, 255, 255)


def set_cell_text(shape, text, size=11.0, color=INK, bold=False, align=PP_ALIGN.LEFT):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.14)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.10)
    tf.margin_bottom = Inches(0.08)
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = "Aptos"
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    return p


def add_text(slide, x, y, w, h, text, size=11.0, color=INK, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.text_frame.vertical_anchor = valign
    set_cell_text(shape, text, size=size, color=color, bold=bold, align=align)
    return shape


def add_header(slide, title, kicker="CGMACROS  /  RESEARCH ARCHITECTURE"):
    add_text(slide, 0.55, 0.25, 12.2, 0.25, kicker, size=10, color=GREEN, bold=True)
    add_text(slide, 0.55, 0.58, 12.2, 0.48, title, size=26, color=FOREST, bold=True)
    add_text(slide, 0.55, 1.12, 12.2, 0.25,
             "Retrospective research PoC  •  45 participants  •  time-indexed two-hour glucose event forecast",
             size=10.5, color=MUTED)


def add_panel(slide, x, y, w, h, title, body, fill=WHITE, accent=GREEN,
              title_size=13.0, body_size=10.2):
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    panel.line.color.rgb = GRID
    panel.line.width = Pt(0.8)
    accent_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                        Inches(x), Inches(y), Inches(0.06), Inches(h))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = accent
    accent_bar.line.fill.background()
    title_box = slide.shapes.add_textbox(Inches(x + 0.16), Inches(y + 0.11),
                                         Inches(w - 0.28), Inches(0.34))
    set_cell_text(title_box, title, size=title_size, color=FOREST, bold=True)
    body_box = slide.shapes.add_textbox(Inches(x + 0.16), Inches(y + 0.48),
                                         Inches(w - 0.28), Inches(h - 0.57))
    set_cell_text(body_box, body, size=body_size, color=INK)
    return panel


def add_connector(slide, x1, y1, x2, y2, color=GREEN, dashed=False, width=1.4):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                           Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = color
    connector.line.width = Pt(width)
    if dashed:
        connector.line.dash_style = 1
    return connector


def add_chevron(slide, x, y, w=0.18, h=0.32, color=GREEN):
    shape = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_footer(slide, text="CGMacros retrospective PoC  •  not for clinical use  •  no diagnosis, treatment, triage, or live monitoring"):
    add_text(slide, 0.55, 7.12, 12.2, 0.18, text, size=8.5, color=MUTED)


def title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PALE
    # Accent block and fine rule echo the existing project presentation.
    block = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), Inches(7.5))
    block.fill.solid()
    block.fill.fore_color.rgb = GREEN
    block.line.fill.background()
    add_text(slide, 0.75, 0.74, 11.8, 0.3, "RESEARCH ARCHITECTURE", size=11, color=GREEN, bold=True)
    add_text(slide, 0.75, 1.22, 11.5, 0.85, "GlycoTwin | CGMacros\nforecast architecture", size=30, color=FOREST, bold=True)
    add_text(slide, 0.78, 2.36, 10.9, 0.52,
             "How static clinical context and multimodal wearable streams become a participant-held-out two-hour glucose forecast.",
             size=15, color=MUTED)

    add_panel(slide, 0.78, 3.35, 3.55, 2.15, "ALIGN", "Join baseline context to each replay timestamp and derive only information available up to t.", fill=MINT)
    add_panel(slide, 4.72, 3.35, 3.55, 2.15, "FUSE", "Combine 41 dynamic rolling/time-series features with 12 baseline clinical features (53 total).")
    add_panel(slide, 8.66, 3.35, 3.55, 2.15, "FORECAST", "Estimate P(Libre glucose at t+120 min ≥ 180 mg/dL), then expose research outputs for review and evaluation.", fill=AMBER_PALE, accent=AMBER)
    add_text(slide, 0.78, 6.42, 11.5, 0.28,
             "Source: docs/architecture.mmd and docs/build_architecture_pdf.py  •  Source cohort: PhysioNet CGMacros v1.0.0",
             size=9, color=MUTED)
    add_footer(slide)


def architecture_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PALE
    add_header(slide, "Architecture: from multimodal data to a forecast")

    # Five input streams, matching architecture.mmd and the PDF generator.
    input_x, input_w, input_h = 0.55, 2.16, 0.70
    input_ys = [1.72, 2.48, 3.24, 4.00, 4.76]
    inputs = [
        ("Baseline clinical profile", "bio.csv: age, gender, BMI, HbA1c, fasting labs, lipids", MINT),
        ("Libre CGM", "Dynamic glucose readings on a 15-minute grid", WHITE),
        ("Heart rate", "Timestamped wearable heart-rate stream", WHITE),
        ("Activity", "Activity level and activity calories", WHITE),
        ("Meals", "Meal events, carbs, calories, and macros", WHITE),
    ]
    for y, (title, body, fill) in zip(input_ys, inputs):
        add_panel(slide, input_x, y, input_w, input_h, title, body, fill=fill, title_size=10.5, body_size=8.3)

    join_x, join_y, join_w, join_h = 3.10, 2.63, 2.25, 1.70
    feat_x, feat_y, feat_w, feat_h = 5.82, 2.63, 1.85, 1.70
    model_x, model_y, model_w, model_h = 8.12, 2.63, 1.78, 1.70
    out_x, out_y, out_w, out_h = 10.35, 2.63, 2.42, 1.70
    add_panel(slide, join_x, join_y, join_w, join_h, "Align and build state",
              "Join by participant and timestamp. Build rolling glucose, heart-rate, activity, meal, lag, and time-of-day features from data available up to t.",
              fill=MINT, title_size=12, body_size=9.1)
    add_panel(slide, feat_x, feat_y, feat_w, feat_h, "Fuse predictors",
              "41 dynamic + 12 baseline features (53 total). Median imputation with missingness indicators.",
              title_size=12, body_size=9.1)
    add_panel(slide, model_x, model_y, model_w, model_h, "Forecast model",
              "HistGradientBoostingClassifier. Participant-grouped cross-validation.",
              title_size=12, body_size=9.1)
    add_panel(slide, out_x, out_y, out_w, out_h, "2-hour output",
              "Estimated probability that Libre glucose at t+120 min is ≥180 mg/dL.",
              fill=AMBER_PALE, accent=AMBER, title_size=12, body_size=9.1)

    # Routed input lines converge into the alignment stage.
    join_mid_y = join_y + join_h / 2
    for y in input_ys:
        add_connector(slide, input_x + input_w, y + input_h / 2, join_x, join_mid_y, color=GREEN, width=1.1)
    add_connector(slide, join_x + join_w, join_mid_y, feat_x, feat_y + feat_h / 2)
    add_connector(slide, feat_x + feat_w, feat_y + feat_h / 2, model_x, model_y + model_h / 2)
    add_connector(slide, model_x + model_w, model_y + model_h / 2, out_x, out_y + out_h / 2)
    for x in [2.82, 5.55, 7.85, 10.08]:
        add_chevron(slide, x, 3.32, color=GREEN)

    # Downstream branches below the main flow.
    ui_x, ui_y, ui_w, ui_h = 3.10, 5.25, 2.25, 0.92
    eval_x, eval_y, eval_w, eval_h = 5.82, 5.25, 1.85, 0.92
    future_x, future_y, future_w, future_h = 8.12, 5.25, 4.65, 0.92
    add_panel(slide, ui_x, ui_y, ui_w, ui_h, "Concept doctor dashboard",
              "Risk-sorted worklist, patient trace, baseline profile, research warning.", fill=MINT, title_size=11, body_size=8.5)
    add_panel(slide, eval_x, eval_y, eval_w, eval_h, "Evaluation",
              "Pooled OOF metrics, HbA1c strata, calibration, participant bootstrap.", title_size=11, body_size=8.5)
    add_panel(slide, future_x, future_y, future_w, future_h, "Future integration only",
              "Sleep and HRV are potential future streams; they are not present in the CGMacros model inputs.", fill=WHITE, accent=AMBER, title_size=11, body_size=8.5)
    add_connector(slide, out_x + out_w * 0.28, out_y + out_h, ui_x + ui_w * 0.50, ui_y, color=GREEN)
    add_connector(slide, out_x + out_w * 0.62, out_y + out_h, eval_x + eval_w * 0.50, eval_y, color=GREEN)
    add_connector(slide, future_x, future_y + future_h / 2, feat_x + feat_w * 0.5, feat_y + feat_h, color=AMBER, dashed=True, width=1.1)
    add_text(slide, 0.55, 6.43, 12.2, 0.27,
             "Dashed path = future integration only. Forecast, dashboard, and evaluation are research outputs—not treatment advice or live monitoring.",
             size=9, color=MUTED)
    add_footer(slide)


def notes_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PALE
    add_header(slide, "Implementation and evaluation notes")
    add_panel(slide, 0.55, 1.72, 3.85, 4.72, "Data contract", "• Baseline: age, gender, BMI, HbA1c, fasting glucose/insulin, and lipids\n\n• Streams: Libre CGM, heart rate, activity, and meals\n\n• Time alignment: participant + timestamp\n\n• State: rolling, lagged, meal, activity, and time-of-day context\n\n• Missing values: median imputation with missingness indicators", fill=MINT, title_size=15, body_size=11)
    add_panel(slide, 4.74, 1.72, 3.85, 4.72, "Model and evaluation", "• 53 predictors: 41 dynamic + 12 baseline\n\n• HistGradientBoostingClassifier\n\n• Five participant-grouped folds\n\n• Pooled out-of-fold metrics and participant-macro views\n\n• HbA1c strata, calibration, and participant-level bootstrap intervals", title_size=15, body_size=11)
    add_panel(slide, 8.93, 1.72, 3.85, 4.72, "Research boundaries", "• Retrospective US cohort; 45 participants\n\n• No Indian cohort or external validation\n\n• No sleep or HRV inputs in this model\n\n• No diagnosis, treatment, triage, or live monitoring\n\n• Dashboard scores and workload rules are exploratory research choices", fill=AMBER_PALE, accent=AMBER, title_size=15, body_size=11)
    add_text(slide, 0.55, 6.62, 12.2, 0.25,
             "Architecture source: docs/architecture.mmd  •  PDF visual reference: docs/build_architecture_pdf.py  •  Dataset: PhysioNet CGMacros v1.0.0",
             size=9, color=MUTED)
    add_footer(slide)


def main():
    # Keep the source architecture close at hand for reproducibility and fail loudly if it is missing.
    if not MMD.exists():
        raise FileNotFoundError(MMD)
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "GlycoTwin CGMacros Forecast Architecture"
    prs.core_properties.subject = "Standalone architecture presentation derived from docs/architecture.mmd"
    prs.core_properties.author = "GlycoTwin"
    title_slide(prs)
    architecture_slide(prs)
    notes_slide(prs)
    prs.save(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
