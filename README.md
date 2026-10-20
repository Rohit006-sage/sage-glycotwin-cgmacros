# GlycoTwin: CGMacros glucose-forecasting proof of concept

> **One-line pitch:** GlycoTwin fuses baseline clinical context with CGM, wearable, activity, and meal signals to forecast whether glucose will reach **180 mg/dL at +120 minutes**.
>
> **Research prototype — retrospective replay only; not for clinical use.** It is not clinically validated, is not a physiological simulator, and must not be used for diagnosis, treatment, triage, live monitoring, or patient care.

## Submission at a glance

| Challenge requirement | Where to review it |
|---|---|
| Team, college, incubator | [Team and submission details](#team-and-submission-details) |
| Project title and use case | This README: [Problem statement](#problem-statement-and-indian-healthcare-relevance) |
| Technical stack | [Technical stack](#technical-stack) |
| AI/ML model or framework details | [AI/ML model and framework details](#aiml-model-and-framework-details) |
| Static + dynamic data fusion | [How the twin works](#how-the-twin-works), [architecture PDF](docs/architecture.pdf), [architecture PPTX](docs/architecture.pptx), [Mermaid source](docs/architecture.mmd) |
| Working adverse-event forecast | [Model and evaluation](docs/EVALUATION.md#evaluation-design-and-model), [aggregate results](docs/results/aggregate_metrics.json) |
| Model card and jury defence | [Model card](docs/MODEL_CARD.md), [jury Q&A](docs/JURY_QA.md) |
| Clinician-facing concept UI | [Dashboard](#dashboard) and `serve_cgmacros_dashboard.py` |
| Video (15–20 minutes) | [Demo video](#demo-video), [recording script](docs/VIDEO_SCRIPT.md); link pending upload |
| Code license and dataset terms | [License and dataset citation](#license-and-dataset-citation), [LICENSE](LICENSE) |
| Architecture diagram | [PDF](docs/architecture.pdf), [PPTX](docs/architecture.pptx), [Mermaid source](docs/architecture.mmd) |
| Presentation | [PPTX](docs/GlycoTwin_CGMacros_Prototype.pptx) and [PDF](docs/GlycoTwin_CGMacros_Prototype.pdf) |
| Data sources | [CGMacros v1.0.0 on PhysioNet](https://doi.org/10.13026/3z8q-x658); raw data are not committed |

## Summary

The twin-oriented prototype gives a clinician a retrospective, risk-ranked view of anonymized CGMacros participants and forecasts a future glucose threshold event. It fuses static EHR-like context with dynamic CGM, wearable, activity, meal, and time-of-day signals, then evaluates first-72-hour per-person adaptation. The cohort is small and from the US; these are retrospective research results, not external or clinical validation.

| Headline result | Existing result |
|---|---:|
| 120-minute mean pooled PR-AUC; event rate | **0.370**; **6.26%** |
| 120-minute pooled ROC-AUC | **0.878** (range **0.855–0.888**) |
| Participant-macro PR-AUC | **0.243** (range **0.241–0.246**) |
| Participant-macro ROC-AUC | **0.741** (range **0.728–0.751**) |
| Same-fold full-fusion pooled PR-AUC / ROC-AUC | **0.382 / 0.888** |
| Same-fold persistence pooled PR-AUC / ROC-AUC | **0.308 / 0.825** |
| Personalization pooled MAE | **20.82 → 18.10 mg/dL** |

These are preserved results from the evaluation artifacts, not new benchmarks from the dashboard enhancement. PR-AUC here is average precision; read it alongside event prevalence and the pooled versus participant-macro definitions in [EVALUATION.md](docs/EVALUATION.md). The personalization MAE has been verified with a leakage-free purged boundary (301 label-crossing samples purged across 45 participants).

## Current implementation and verification status

The project has a working local inference path and a retrospective replay API, but is **not yet submission-ready without the corrections below**. No clinical validation or winning-position claim is made.

| Area | Current evidence | Remaining gap |
|---|---|---|
| Prediction workflow | Existing fitted-cache inference and CLI checked; cache integrity and schema verified | Match deployment environment to trusted cache manifest |
| Automated checks | Locked Python 3.11 environment verified; metric, documentation, and full suite: **55 passed** in 5.9s | GitHub Actions has not run yet because there is no remote configured |
| Forecast-versus-observed replay | API hides the future outcome until explicit `+120` advancement; verified live via local replay server across all 45 participants | Validate screen recording during walkthrough |
| Timeline | Missing values stay gaps, lines break cleanly across gaps > 20 min (`segmentSeries`), isolated readings render as dots. Verified via Node tests and live asset serving | Capture and review the five dashboard screenshots before recording |
| Fold provenance | Primary and optional multi-horizon sources are labeled separately; optional source resolves its own fold metadata with fallback to `revised_full_metrics.json` | Re-run full 57m `revised_evaluation.py` if CSV column is desired without metadata fallback |
| Personalization | Leakage-free rerun completed with purged boundary: 301 crossing labels purged across 45 participants; pooled MAE improves 20.82 → 18.10 mg/dL (macro MAE: 21.07 → 18.17 mg/dL). `aggregate_metrics.json` and SVG updated | Individual comparison in dashboard remains withheld |
| Documentation | Schema and reported personalization metrics synchronized; the report consistency script checks public text and slide values against `aggregate_metrics.json` | Independent external validation remains unavailable; see the locked evaluation plan |
| Submission | README, model card, evaluation details, presentation, and video script are prepared | Add the 15–20 minute video URL and dashboard screenshots |

The local locked-environment check passed, including the aggregate-metric and documentation checks. GitHub Actions has not run because no repository remote is configured. These checks do not provide external or clinical validation.

## Team and submission details

- **Team name:** Sage
- **Team leader:** Manda Rohit Vardhan
- **Members and roles:** Manda Rohit Vardhan — Project Lead and Developer
- **College:** Amrita Vishwa Vidyapeetham, Hyderabad
- **Incubator:** None
- **Recommended repository name:** `sage-glycotwin-cgmacros`
- **Submission repository:** [https://github.com/Rohit006-sage/sage-glycotwin-cgmacros](https://github.com/Rohit006-sage/sage-glycotwin-cgmacros)

The team leader's phone number and email belong in the submission form only; do not publish them in this README.

## Demo video

**Video link:** To be added after uploading the unlisted YouTube demo (submission requirement: 15–20 minutes)

The recording should cover the problem and India context, data licensing, twin loop, model/evaluation, results and limitations, and a live or recorded dashboard walkthrough. Use the [timed video script](docs/VIDEO_SCRIPT.md). The video must be accessible without additional permission beyond the unlisted YouTube link.

## Problem statement and Indian healthcare relevance

Type 2 diabetes is an important Indian health priority. ICMR-INDIAB estimated 101 million people in India had diabetes in 2021, with weighted prevalence of 11.4%. This project asks whether baseline clinical context plus recent continuous glucose monitoring (CGM), heart rate, activity, and meal information can forecast elevated glucose. The two-hour horizon is a prototype follow-up window for clinician review, not a clinically established alert interval. The event is CGM glucose at least 180 mg/dL at the selected future horizon, a domain-chosen upper-range boundary rather than a threshold tuned to the observed event rate. See the [ICMR-INDIAB study](https://doi.org/10.1016/S2213-8587(23)00119-5) and [ADA Standards of Care 2026, Section 6](https://diabetesjournals.org/care/article/49/Supplement_1/S132/163927/6-Glycemic-Goals-Hypoglycemia-and-Hyperglycemic).

India needs locally representative, consented longitudinal data spanning diverse regions, care settings, diets, socioeconomic contexts, and device access patterns. CGMacros is a US cohort with 45 participants, so its learned risk levels, calibration, and performance cannot be assumed to transfer to India; Indian model development would require local calibration and prospective multi-site validation. The project has no independent cohort for external validation; see the [external validation plan](docs/EXTERNAL_VALIDATION_PLAN.md).

## How the twin works

1. **Static EHR-like layer:** age, sex, BMI, HbA1c, fasting glucose, fasting insulin, and lipid measurements provide baseline context.
2. **Dynamic layer:** time-aligned CGM, heart rate, activity, meal, and time-of-day signals describe the recent state.
3. **Fusion:** rolling glucose statistics, time-above-range measures, meal timing, a heuristic carbohydrate-on-board proxy, wearable/activity features, and baseline context are joined at each forecast timestamp.
4. **Per-person adaptation experiment:** the intended adaptation/training window is the first 72 hours. A leakage-free rerun with a purged boundary (301 label-crossing samples purged across all 45 participants) achieved pooled MAE **20.82 → 18.10 mg/dL** (macro MAE: **21.07 → 18.17 mg/dL**). Eligibility and caveats are documented in [Full evaluation details](docs/EVALUATION.md#chronological-within-person-evaluation).
5. **Forecast:** scikit-learn `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` produce the two-hour event probability and glucose estimate.
6. **Clinician concept view:** the replay dashboard shows a risk-ranked worklist, selected glucose trace, forecast score, context, and baseline profile.

This is **not a physiological simulator**: it does not simulate physiology, treatment response, or causal mechanisms, and it is not a validated clinical digital twin.

## Technical stack

- **Language/runtime:** Python 3.11.15
- **Data and scientific computing:** pandas, NumPy, SciPy
- **Machine learning:** scikit-learn, including `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, grouped cross-validation, calibration, and participant-level bootstrap analysis
- **Replay interface:** semantic HTML, CSS, and vanilla JavaScript served by a local Python HTTP server
- **Testing:** pytest and Node.js tests for pure replay helpers
- **Reproducibility and CI:** pinned Python dependencies, PowerShell verification script, GitHub Actions, and documentation/link checks

## AI/ML model and framework details

The prototype performs multimodal feature fusion at 15-minute forecast timestamps. Static baseline clinical context is joined with recent CGM, heart-rate, activity, meal, lag, rolling, and time-of-day features. The classifier estimates the probability that Libre glucose will be at least **180 mg/dL at +120 minutes**. The regressor estimates the future glucose value at the same horizon. Evaluation uses participant-grouped held-out folds, pooled and participant-macro metrics, calibration analysis, subgroup summaries, and a chronological first-72-hour adaptation experiment with a purged label-availability boundary. All reported results are retrospective research evidence, not clinical validation.

## Dashboard

The local dashboard is a retrospective replay of seed-42 held-out predictions. It is explicitly not a live monitor or clinical tool. It provides a dataset-ID participant selector, eligible timestamp navigation, an exact-timestamp worklist, feature-snapshot plots for glucose/heart rate/activity/meal summaries, a forecast-only state, and an explicit advance-to-observed-outcome step. The primary source shows the held-out event score; the optional seed-42 multi-horizon source also shows its paired regression estimate and reports its own verified fold provenance. The API and numeric cards preserve missing states and use the saved exact-horizon target, not a nearest observation. Canvas rendering cleanly preserves null gaps and breaks lines across intervals exceeding 20 minutes (`segmentSeries`). Dataset IDs are pseudonymous, not a guarantee of anonymity. See the [click-by-click jury demo](docs/DEMO_SCRIPT.md).

The dashboard does **not** display individual personalization comparisons. While the aggregate evaluation has been verified with a purged boundary (**20.82 → 18.10 mg/dL** pooled MAE), individual paired adaptation replay is withheld from the UI to avoid presenting exploratory research as a clinical tool. The complete aggregate results are detailed in [Full evaluation details](docs/EVALUATION.md).

Screenshots to capture from the local replay are specified in [docs/screenshots/README.md](docs/screenshots/README.md). They are intentionally not generated from a clean checkout because the replay outputs are git-ignored:

1. `docs/screenshots/01-risk-ranked-worklist.png`
2. `docs/screenshots/02-selected-glucose-trace.png`
3. `docs/screenshots/03-forecast-context.png`
4. `docs/screenshots/04-static-ehr-summary.png`
5. `docs/screenshots/05-empty-state-or-banner.png`

## Architecture and presentation

- [Architecture diagram (PDF)](docs/architecture.pdf)
- [Architecture presentation (PPTX)](docs/architecture.pptx)
- [Editable architecture source](docs/architecture.mmd)
- [Project presentation (PPTX)](docs/GlycoTwin_CGMacros_Prototype.pptx)
- [Project presentation (PDF)](docs/GlycoTwin_CGMacros_Prototype.pdf)

## Public access and submission files

The intended final submission package contains this README, the MIT project license, the architecture PDF/PPTX/source, the project presentation PDF/PPTX, the evaluation and model-card documents, and the unlisted YouTube demo link. Before submission, verify that the GitHub repository and all linked files open while logged out and without additional permissions. Raw CGMacros data, participant-level files, and per-sample predictions are not part of the public repository.

## Full evaluation details

The complete existing evaluation write-up, including every result table, ablation, subgroup bootstrap interval, alert workload analysis, chronological personalization result, caveat, and reproduction command is preserved in [docs/EVALUATION.md](docs/EVALUATION.md). Do not interpret any result as external validation or clinical benefit.

## Quick start

### Run one prediction

This inference-only demo uses the existing seed-42 outer-fold fits. Export the verified fold cache once from local ignored data, then route a participant to the fold where that participant was held out.

#### Step 1 — export the verified cache once

PowerShell:

```powershell
python export_demo_cache.py
```

bash:

```bash
python3 export_demo_cache.py
```

The exporter compares regenerated predictions with the stored seed-42 held-out scores before writing anything. It writes only under `data/processed/models/`, which is git-ignored.

The demo cache uses the seed-42 multi-horizon regression folds from `StratifiedGroupKFold`, which differ from the five primary repeated-CV assignments behind the headline metrics, so the demo shows the same method but not row-for-row the headline folds.

#### Step 2 — run one held-out replay prediction

PowerShell:

```powershell
python predict_demo.py `
  --participant 002 `
  --timestamp 2019-11-16T10:15 `
  --data-dir "C:\path\to\data\raw\cgmacros\<dataset>\extracted\CGMacros" `
  --manifest data/processed/models/seed42_manifest.json `
  --cache-dir data/processed/models `
  --trust-cache
```

bash:

```bash
python3 predict_demo.py \
  --participant 002 \
  --timestamp 2019-11-16T10:15 \
  --data-dir "data/raw/cgmacros/<dataset>/extracted/CGMacros" \
  --manifest data/processed/models/seed42_manifest.json \
  --cache-dir data/processed/models \
  --trust-cache
```

Example output format only; values below are placeholders, not participant results:

```text
Research prototype — retrospective replay only; not for clinical use.
{
  "participant_id": "[ANONYMIZED ID]",
  "timestamp": "[LOCAL ISO TIMESTAMP]",
  "probability_glucose_at_or_above_threshold": "[PLACEHOLDER]",
  "regression_glucose_mg_dl": "[PLACEHOLDER]",
  "context": {
    "trend": "[rising|falling|stable|unavailable]",
    "carbs_last_120m": "[PLACEHOLDER]",
    "activity_level": "[PLACEHOLDER]"
  }
}
```

The CLI is intentionally inference-only: it does not retrain or retune the model, generate new metrics, or use saved out-of-fold scores as fresh predictions. It prints the selected held-out fold and the retrospective replay warning.

### Reproducibility and data safety

- Use Python 3.11.15 (the version in `.python-version`) and create a clean virtual environment:

  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements-lock.txt
  ```

- Obtain CGMacros v1.0.0 separately from PhysioNet and extract it under the ignored `data/raw/cgmacros/` path. Raw participant records are not committed.
- Run `powershell -ExecutionPolicy Bypass -File .\scripts\verify_repro.ps1`. It checks the Python version and installed dependencies, compares published personalization figures with the aggregate JSON, and runs the complete test suite. The full pipeline contract test requires the local dataset.
- Direct runtime dependencies are listed in `requirements.txt`; test tooling is listed in `requirements-dev.txt`. `requirements-lock.txt` freezes the complete verified Python 3.11 environment used for local checks. The supported Python range is declared in `pyproject.toml`.
- Raw CGMacros records, processed participant-level rows, per-sample predictions, and model caches remain under ignored paths and are not part of the public artifact.
- The complete evaluation commands and their data requirements are documented in [Full evaluation details](docs/EVALUATION.md#run).

This local verification is not a clean-checkout certification until it passes in a newly created environment after separately obtaining the licensed dataset. It does not reproduce the full evaluation; see [Full evaluation details](docs/EVALUATION.md#run) for the longer analysis command.

For the synthetic inference, export, dashboard, adaptation split, and ODE tests without downloading CGMacros:

```powershell
python -m pip install pytest==8.3.4
python -m pytest -q tests/test_predict_demo.py tests/test_export_demo_cache.py tests/test_dashboard.py tests/test_twin_ode.py tests/test_adaptation_split.py tests/test_replay_js.py
```

Install the runtime dependencies first, preferably in a virtual environment. The original `tests/test_pipeline.py` imports the local dataset and includes the full-data participant/sample/event count check; it is intentionally excluded from the data-free command above. These commands run tests, not evaluation or model training.

### 5-minute demo

The dashboard command is:

```powershell
python serve_cgmacros_dashboard.py
```

Open <http://127.0.0.1:8765>. The dashboard replays locally generated outputs; see the [screenshots](docs/screenshots/README.md) and [demo video script](docs/VIDEO_SCRIPT.md). It is not live monitoring or a clinical tool.

1. Select a participant and an eligible historical timestamp.
2. Start with the **Primary seed-42 OOF** source and review the score and numeric context cards; the future outcome is hidden.
3. If available, select the optional paired multi-horizon source to inspect its regression estimate and verified fold provenance.
4. Click **Advance to observed +120 minutes** to reveal the exact saved outcome and individual absolute regression error when available.
5. Click **Reset to forecast time**. Changing participant, timestamp, or source also resets the reveal state.
6. Explain missing states, the uncalibrated score, and the withheld personalization comparison. An individual replay does not establish aggregate performance.

The server binds to loopback, reads local replay CSVs, and does not retrain, deserialize models, or write new evaluation artifacts. A clean checkout has no replay CSVs: generate them using the documented evaluation workflow first. Timeline canvas rendering cleanly preserves null gaps and breaks lines across recording jumps (>20 min) via `segmentSeries`.

### Full reproduction (~57 minutes)

Download CGMacros v1.0.0 separately from [PhysioNet](https://doi.org/10.13026/3z8q-x658), place extracted files under `data/raw/cgmacros/`, install [requirements.txt](requirements.txt), and run the commands in [Full evaluation details](docs/EVALUATION.md#run). Raw data and detailed predictions remain local and git-ignored.

## Limitations and deployment path

- Only 45 participants are available, and results vary across participant assignments. Pooled, participant-macro, and chronological estimates answer different questions.
- The data are from the United States. India-specific model development requires representative Indian data, local calibration, and prospective multi-site validation.
- This is not a physiological simulator and does not estimate causal treatment response; medication timing and dose are not modeled. Sleep and HRV are absent from the current streams.
- Episode merge and alert cooldown values in the retrospective workload analysis are exploratory. They need clinician input, prospective validation, and a defined follow-up action before any alert workflow is considered.
- No external validation, prospective study, fairness assessment, or clinical utility study has been performed. Do not connect this prototype to live devices, patient records, or care workflows.
- A deployment path requires ethics and privacy review, secure data governance, prospective validation, clinician workflow evaluation, monitoring, and a clear escalation protocol. Assess India's [Digital Personal Data Protection Act 2023](https://www.meity.gov.in/static/uploads/2024/02/Digital-Personal-Data-Protection-Act-2023.pdf). HIPAA applies only where the responsible entities and data fall within its scope; see [HHS covered entities and business associates](https://www.hhs.gov/hipaa/for-professionals/covered-entities/index.html).

## License and dataset citation

Project code is licensed under the [MIT License](LICENSE). This is separate from the CGMacros dataset license, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Review dataset terms before distributing data or derived materials.

Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). *CGMacros: a scientific dataset for personalized nutrition and diet monitoring* (version 1.0.0). PhysioNet. [doi:10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658).
