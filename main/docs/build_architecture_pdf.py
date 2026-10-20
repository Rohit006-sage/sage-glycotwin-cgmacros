from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

OUTPUT = "docs/architecture.pdf"
PAGE_W, PAGE_H = landscape(letter)
FOREST = colors.HexColor("#0A3B2C")
GREEN = colors.HexColor("#16745B")
MINT = colors.HexColor("#EAF3EE")
PALE = colors.HexColor("#F5F7F6")
AMBER = colors.HexColor("#D28A2E")
INK = colors.HexColor("#19342B")
MUTED = colors.HexColor("#61776E")
WHITE = colors.white


def wrapped(c, text, x, y, width, font="Helvetica", size=9, leading=12, color=INK):
    c.setFillColor(color)
    c.setFont(font, size)
    words = text.split()
    lines, line = [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and stringWidth(candidate, font, size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def box(c, x, y, w, h, title, body, fill=WHITE, accent=GREEN, title_size=10, body_size=8.2):
    c.setFillColor(fill)
    c.setStrokeColor(colors.HexColor("#CBD9D0"))
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    c.setFillColor(accent)
    c.roundRect(x, y, 5, h, 2, fill=1, stroke=0)
    c.setFillColor(FOREST)
    c.setFont("Helvetica-Bold", title_size)
    c.drawString(x + 14, y + h - 19, title)
    wrapped(c, body, x + 14, y + h - 36, w - 26, size=body_size, leading=10.4)


def arrow(c, x1, y1, x2, y2, color=GREEN, dashed=False):
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.5)
    if dashed:
        c.setDash(4, 3)
    c.line(x1, y1, x2, y2)
    c.setDash()
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    size, wing = 7, 3.5
    bx, by = x2 - size * ux, y2 - size * uy
    c.line(x2, y2, bx - wing * uy, by + wing * ux)
    c.line(x2, y2, bx + wing * uy, by - wing * ux)


def routed_up(c, x1, y1, x2, y2, route_y):
    c.setStrokeColor(GREEN)
    c.setLineWidth(1.5)
    c.line(x1, y1, x1, route_y)
    c.line(x1, route_y, x2, route_y)
    c.line(x2, route_y, x2, y2)
    arrow(c, x2, route_y, x2, y2)
    c.restoreState()


def main():
    c = canvas.Canvas(OUTPUT, pagesize=landscape(letter))
    c.setTitle("GlycoTwin CGMacros Forecast Architecture")
    c.setFillColor(PALE)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(FOREST)
    c.setFont("Helvetica-Bold", 21)
    c.drawString(34, PAGE_H - 38, "GlycoTwin | CGMacros forecast architecture")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(34, PAGE_H - 55, "Retrospective research PoC | 45 participants | time-indexed two-hour glucose event forecast")

    # Input streams
    input_x, input_w, input_h = 35, 175, 66
    input_ys = [PAGE_H - 148, PAGE_H - 224, PAGE_H - 300, PAGE_H - 376, PAGE_H - 452]
    inputs = [
        ("Baseline clinical profile", "bio.csv: age, gender, BMI, HbA1c, fasting glucose/insulin, lipids", MINT),
        ("Libre CGM", "Dynamic glucose readings on a 15-minute grid", WHITE),
        ("Heart rate", "Timestamped wearable heart-rate stream", WHITE),
        ("Activity", "Activity level and activity calories", WHITE),
        ("Meals", "Meal events and recent carbohydrate/macronutrient totals", WHITE),
    ]
    for y, (title, body, fill) in zip(input_ys, inputs):
        box(c, input_x, y, input_w, input_h, title, body, fill=fill)

    join_x, join_y, join_w, join_h = 255, PAGE_H - 304, 168, 112
    box(c, join_x, join_y, join_w, join_h, "Align and build state",
        "Join baseline clinical context by participant. At each timestamp, create rolling glucose, heart-rate, activity, meal, lag and time-of-day features from data available up to that time.", fill=MINT)
    for y in input_ys:
        arrow(c, input_x + input_w, y + input_h / 2, join_x, join_y + join_h / 2)

    feat_x, feat_y, feat_w, feat_h = 457, PAGE_H - 304, 144, 112
    box(c, feat_x, feat_y, feat_w, feat_h, "Fuse predictors",
        "41 dynamic features + 12 baseline features (53 total). Median imputation with missingness indicators.", fill=WHITE)
    arrow(c, join_x + join_w, join_y + join_h / 2, feat_x, feat_y + feat_h / 2)

    model_x, model_y, model_w, model_h = 633, PAGE_H - 304, 130, 112
    box(c, model_x, model_y, model_w, model_h, "Forecast model",
        "HistGradientBoostingClassifier. Evaluated with five participant-grouped folds; scores shown are out-of-fold.", fill=WHITE)
    arrow(c, feat_x + feat_w, feat_y + feat_h / 2, model_x, model_y + model_h / 2)

    forecast_x, forecast_y, forecast_w, forecast_h = 633, 92, 130, 83
    box(c, forecast_x, forecast_y, forecast_w, forecast_h, "2-hour output",
        "Estimated probability that Libre glucose at t+120 min is >=180 mg/dL.", fill=colors.HexColor("#FFF5E5"), accent=AMBER)
    arrow(c, model_x + model_w / 2, model_y, forecast_x + forecast_w / 2, forecast_y + forecast_h)

    ui_x, ui_y, ui_w, ui_h = 255, 92, 168, 83
    box(c, ui_x, ui_y, ui_w, ui_h, "Concept dashboard",
        "Risk-sorted worklist, selected patient history, baseline profile and research-only warning.", fill=MINT)
    routed_up(c, forecast_x, forecast_y + forecast_h / 2, ui_x + ui_w / 2, ui_y + ui_h, 200)

    eval_x, eval_y, eval_w, eval_h = 457, 92, 144, 83
    box(c, eval_x, eval_y, eval_w, eval_h, "Evaluation outputs",
        "Pooled out-of-fold metrics, HbA1c strata, reliability plot and participant-level bootstrap intervals.", fill=WHITE)
    routed_up(c, forecast_x + forecast_w / 2, forecast_y + forecast_h, eval_x + eval_w / 2, eval_y + eval_h, 225)

    # Limitations note and source
    c.setFillColor(AMBER)
    c.rect(34, 48, PAGE_W - 68, 2, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(34, 33, "NOT A CLINICAL SYSTEM")
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED)
    c.drawString(156, 33, "CGMacros is not an Indian cohort; sleep and HRV are not inputs; no treatment advice or live monitoring.")
    c.drawRightString(PAGE_W - 34, 18, "Source: PhysioNet CGMacros v1.0.0 | README lists data and clinical references")
    c.save()


if __name__ == "__main__":
    main()

