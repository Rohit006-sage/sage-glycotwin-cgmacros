# Five-minute jury demo: CGMacros replay dashboard

Jury discussion reference: [Jury Q&A](JURY_QA.md).

This exact script demonstrates the **research prototype** using locally generated held-out replay outputs. Say “retrospective replay” throughout. Do not describe the page as live monitoring, triage, diagnosis, treatment, or a clinical alert system.

## Setup

1. In PowerShell, from the project folder, run `python .\serve_cgmacros_dashboard.py`.
2. Wait for `Opening local-only CGMacros replay dashboard at http://127.0.0.1:8765`.
3. Open `http://127.0.0.1:8765` and keep the terminal visible in a second window.

If the CLI prediction path is shown alongside the dashboard, use participant `002` at `2019-11-16T10:15` with the verified local seed-42 fold cache; this is a retrospective held-out replay, not a live prediction.

## Click-by-click (5:00)

| Time | Click / pointer action | Exact narration |
|---|---|---|
| 0:00–0:30 | Point to the header and research banner. | “This is a local CGMacros retrospective research replay. It is not live monitoring, not clinically validated, and has no alert cutoff. IDs and dates are dataset IDs and time-shifted.” |
| 0:30–1:15 | Scroll to the **Risk-ranked retrospective review list**. Open the **Participant** dropdown and click the first participant. | “The list is ranked by the held-out seed-42 score at this replay position. This ordering is for review, not triage.” |
| 1:15–2:00 | Move the eligible timestamp slider. | “Only aligned feature snapshots at or before this forecast time are shown. The outcome is hidden. Worklist entries require the exact shared timestamp; no nearest replacement is used.” |
| 2:00–2:45 | Select the optional multi-horizon source if available; inspect provenance. | “This separate seed-42 StratifiedGroupKFold demo source pairs its classifier score with its regression estimate. It is not row-for-row the primary headline evaluation.” |
| 2:45–3:30 | Click **Advance to observed +120 minutes**, then **Reset to forecast time**. | “The exact historical target is now revealed, with retrospective error if a regression estimate is available. A single prediction does not establish model performance; missing targets remain unavailable.” |
| 3:30–4:10 | Point to heart rate, activity, and meal context cards. | “These cards show the time-aligned wearable and activity context plus carbs recorded in the prior two hours. They are inputs for review, not recommendations.” |
| 4:10–4:40 | Point to **Static EHR summary** and scan the baseline fields. | “This is baseline EHR-like context recorded at study start, shown alongside the dynamic replay. It is not a live chart.” |
| 4:40–5:00 | Point back to the banner and footer. | “The cohort is small and from the US; this remains a research prototype. External validation, prospective evaluation, and clinical utility work are still required.” |

The synchronized plots show available model-input snapshots and rolling meal summaries, not raw sensor feeds. Keep the data-quality note visible: a zero rolling meal summary does not establish complete meal logging. Aggregate personalization was rerun with a purged label-availability boundary, but the individual personalization panel remains unavailable until that comparison is validated for interactive replay; do not describe historical aggregate metrics as a live comparison.

## Fallback

If startup prints `Dashboard replay outputs are missing; the dashboard was not started`, do not retry during the jury demo. Use the prepared screenshots or recorded walkthrough in this order: risk-ranked list, selected trace, forecast/context, static EHR summary, and banner/empty state. Keep the narration above and say: “The local replay outputs are unavailable because they are git-ignored; this is the saved retrospective view of the same interface.”

If no screenshots or recording are available, show `README.md` and `docs/EVALUATION.md` sections for the dashboard, caveats, and aggregate metrics only. Do not open or share raw CGMacros files, participant-level CSVs, or per-sample predictions.
