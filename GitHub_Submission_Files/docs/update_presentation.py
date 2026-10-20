from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


SRC = "docs/GlycoTwin_CGMacros_Prototype.pptx"


def add_slide(prs, title, kicker, sections):
    # This deck exposes only its custom title/content layouts; clear any
    # placeholders from the first available layout before adding our panels.
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    for shape in list(slide.shapes):
        if getattr(shape, "is_placeholder", False):
            shape._element.getparent().remove(shape._element)
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = RGBColor(245, 247, 246)

    tx = slide.shapes.add_textbox(Inches(0.55), Inches(0.28), Inches(12.2), Inches(0.35))
    p = tx.text_frame.paragraphs[0]
    p.text = "GLYCOTWIN  /  RESEARCH PROTOTYPE"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(22, 116, 91)

    h = slide.shapes.add_textbox(Inches(0.55), Inches(0.7), Inches(12.2), Inches(0.6))
    p = h.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = RGBColor(10, 59, 44)

    k = slide.shapes.add_textbox(Inches(0.55), Inches(1.28), Inches(12.2), Inches(0.3))
    p = k.text_frame.paragraphs[0]
    p.text = kicker
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(97, 119, 110)

    n = len(sections)
    gap = 0.22
    w = (12.2 - gap * (n - 1)) / n
    for i, (heading, body) in enumerate(sections):
        x = 0.55 + i * (w + gap)
        box = slide.shapes.add_textbox(Inches(x), Inches(1.82), Inches(w), Inches(5.0))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(255, 255, 255)
        box.line.color.rgb = RGBColor(203, 217, 208)
        tf = box.text_frame
        tf.clear()
        tf.margin_left = Inches(0.18)
        tf.margin_right = Inches(0.15)
        tf.margin_top = Inches(0.15)
        p = tf.paragraphs[0]
        p.text = heading
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = RGBColor(10, 59, 44)
        for line in body:
            p = tf.add_paragraph()
            p.text = line
            p.level = 0
            p.font.size = Pt(11)
            p.font.color.rgb = RGBColor(25, 52, 43)
            p.space_after = Pt(8)

    foot = slide.shapes.add_textbox(Inches(0.55), Inches(7.08), Inches(12.2), Inches(0.25))
    p = foot.text_frame.paragraphs[0]
    p.text = "CGMacros retrospective PoC  •  not for clinical use  •  no diagnosis, treatment, triage, or live monitoring"
    p.font.size = Pt(9)
    p.font.color.rgb = RGBColor(97, 119, 110)
    return slide


def main():
    prs = Presentation(SRC)
    submission_replacements = {
        "[TEAM NAME]": "Team name: Sage",
        "[TEAM LEAD NAME]": "Team lead: Manda Rohit Vardhan",
        "[ADD NAMES AND ROLES]": "Members and roles: Manda Rohit Vardhan — Project Lead and Developer",
        "[COLLEGE / INCUBATOR OR NONE]": "College: Amrita Vishwa Vidyapeetham, Hyderabad; incubator: None",
        "[SHAREABLE 20+ MINUTE VIDEO LINK]": "Unlisted YouTube demo link: to be added (15–20 minutes)",
        "[PUBLIC REPOSITORY URL]": "Public GitHub repository: github.com/Rohit006-sage/sage-glycotwin-cgmacros",
    }
    generated_titles = {
        "Method and fair comparison",
        "Limitations and path to deployment",
        "Conclusion and submission placeholders",
        "Conclusion and submission details",
    }
    remove_slide_ids = []
    for slide_id, slide in zip(list(prs.slides._sldIdLst), list(prs.slides)):
        slide_text = " ".join(
            getattr(shape, "text", "")
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False)
        )
        if any(title in slide_text for title in generated_titles):
            remove_slide_ids.append(slide_id)
    for slide_id in remove_slide_ids:
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)
    # Keep the existing architecture slide synchronized with docs/architecture.mmd
    # and docs/build_architecture_pdf.py (41 dynamic + 12 baseline = 53 total).
    for slide in prs.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            text_frame = getattr(shape, "text_frame", None)
            if text_frame is None:
                continue
            for paragraph in text_frame.paragraphs:
                for run in paragraph.runs:
                    for old, new in submission_replacements.items():
                        run.text = run.text.replace(old, new)
                    run.text = run.text.replace(
                        "29 dynamic + 12 baseline features (41 total)",
                        "41 dynamic + 12 baseline features (53 total)",
                    ).replace(
                        "29 dynamic + 12 baseline features",
                        "41 dynamic + 12 baseline features (53 total)",
                    )
    add_slide(
        prs,
        "Method and fair comparison",
        "The evaluation separates participant generalization, personalization, and exploratory workload decisions.",
        [
            ("Repeated grouped CV", [
                "Five distinct participant assignments; outer participants remain held out.",
                "Inner participant-grouped search; cutoffs selected from training-participant OOF scores.",
                "Report pooled and participant-macro metrics because they answer different questions.",
            ]),
            ("Personalization", [
                "Fixed first 72 hours for adaptation/training; 43/45 participants qualify (≥200 early labeled intervals).",
                "Early-data adaptation: pooled/macro PR-AUC 0.523/0.251; population-only 0.374/0.240.",
                "Regression pooled MAE: 20.82 → 18.10 mg/dL (purged boundary, 301 crossing labels removed).",
            ]),
            ("Same-fold + workload", [
                "Seed-42 held-out comparison: full-fusion HGB PR-AUC 0.382; reduced five-variable EHR + sensors 0.400; persistence 0.308; trend 0.226.",
                "Top-5% episode cutoffs: 199/391 (30m), 197/389 (60m), 122/384 (120m).",
                "30-minute merge and 60-minute cooldown are exploratory; false alerts 0.25–1.14 per approximate participant-day.",
            ]),
        ],
    )
    add_slide(
        prs,
        "Limitations and path to deployment",
        "A research workflow can inform the next study only after local validation, governance, and clinician-defined action.",
        [
            ("Current limits", [
                "45-participant US cohort; no Indian cohort, external validation, prospective study, fairness assessment, or clinical utility evaluation.",
                "Not a physiological simulator; medication timing/dose, treatment response, sleep, and HRV are not modeled.",
                "Episode merge, cooldown, thresholds, and dashboard scores are retrospective research choices.",
            ]),
            ("Privacy and governance", [
                "Assess India Digital Personal Data Protection Act (DPDP) obligations for any local data work.",
                "HIPAA applies only when responsible entities and data fall within its scope; obtain qualified review.",
                "Use consented, governed, de-identified data; do not ship raw records or participant-level derived outputs.",
            ]),
            ("Deployment path", [
                "Representative Indian data → local calibration/retraining → prospective multi-site validation.",
                "Define clinician action, escalation, monitoring, audit, and workload targets before any alert workflow.",
                "Only then evaluate secure integration with approved devices and records; this prototype must not connect to care workflows.",
            ]),
        ],
    )
    add_slide(
        prs,
        "Conclusion and submission details",
        "GlycoTwin demonstrates a reproducible multimodal forecasting workflow; the evidence supports further research, not clinical use.",
        [
            ("What the prototype shows", [
                "Static baseline context plus CGM, heart rate, activity, meals, and time-of-day can be fused at forecast timestamps.",
                "Grouped evaluation, same-fold baselines, chronological personalization, calibration, and episode workload expose trade-offs.",
                "The architecture matches the pipeline: align → fuse 41 dynamic + 12 baseline features (53 total) → HGB → forecast → dashboard/evaluation.",
            ]),
            ("Team", [
                "Team name: Sage",
                "Team lead and member: Manda Rohit Vardhan — Project Lead and Developer",
                "College: Amrita Vishwa Vidyapeetham, Hyderabad",
                "Incubator: None",
            ]),
            ("Submission", [
                "Demo video: unlisted YouTube link to be added (15–20 minutes)",
                "Repository: github.com/Rohit006-sage/sage-glycotwin-cgmacros",
                "Dataset citation: CGMacros v1.0.0, PhysioNet, DOI 10.13026/3z8q-x658.",
                "Retain the research-only notice and dataset/privacy review before sharing.",
            ]),
        ],
    )
    prs.save(SRC)


if __name__ == "__main__":
    main()
