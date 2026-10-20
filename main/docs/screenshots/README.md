# Dashboard screenshot capture kit

These five screenshots are the exact capture set for the jury fallback. Capture them only after generating the local held-out replay outputs and starting the dashboard. Do not commit raw CGMacros files, participant-level CSVs, or per-sample predictions.

## Capture procedure

1. Download CGMacros separately under the ignored `data/raw/cgmacros/` path and run the existing reproduction command in [EVALUATION.md](../EVALUATION.md#run).
2. Start the local replay with `python .\serve_cgmacros_dashboard.py`.
3. Open `http://127.0.0.1:8765` in a maximized browser window. Use the browser's built-in screenshot tool or OS snipping tool; do not use a data export.
4. Save PNGs with the filenames below. Use only the anonymized participant IDs already rendered by the dashboard, and keep the research banner visible where specified.
5. Review each image at 100% before sharing. Remove browser tabs, local filesystem paths, terminal output, and any raw-data preview.

## Required filenames and framing

| Filename | What must be visible |
|---|---|
| `01-risk-ranked-worklist.png` | Research-prototype banner and the risk-ranked retrospective review list |
| `02-selected-glucose-trace.png` | Selected anonymized participant, historical glucose trace, and time slider |
| `03-forecast-context.png` | +120-minute retrospective probability plus CGM, heart-rate, activity, and meal context |
| `04-static-ehr-summary.png` | Static EHR-like summary including HbA1c, BMI, and baseline fields |
| `05-empty-state-or-banner.png` | The research-only retrospective warning and either the empty state or the dashboard footer |

## Suggested capture sequence

Use anonymized participant `001` for the selected-person views. Start with the slider at its initial position for the worklist and trace, then move it approximately one quarter to the right for the forecast/context and static-profile views. Do not record or publish the underlying timestamp, source rows, browser developer tools, terminal output, or local filesystem paths.

1. `01-risk-ranked-worklist.png`: participant selector on `001`, slider at its initial position, full worklist and banner visible.
2. `02-selected-glucose-trace.png`: participant `001`, slider at its initial position, selected retrospective trace visible.
3. `03-forecast-context.png`: participant `001`, slider approximately one quarter right, forecast probability and context cards visible.
4. `04-static-ehr-summary.png`: participant `001`, same quarter-right slider position, static EHR summary visible.
5. `05-empty-state-or-banner.png`: keep the banner and footer visible; use the styled empty state only if the dashboard is intentionally started without replay outputs.

The PNGs are submission assets, not model outputs. If they are not captured before packaging, use the recorded dashboard walkthrough instead and keep this instruction file as the reproducible capture specification.
