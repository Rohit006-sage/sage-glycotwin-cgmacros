# GlycoTwin model card

**Research prototype only; not clinically validated.** Intended for retrospective clinician-review research concerning established type 2 diabetes (T2D) follow-up. Excluded uses: diagnosis, treatment, triage, live monitoring, and patient care; do not connect to live devices, patient records, or care workflows.

## Data, inputs, and model

- **Data:** CGMacros v1.0.0, 45 US participants; the mixed cohort is not an Indian cohort or a chart-confirmed T2D cohort. HbA1c strata are analytic categories. Dataset citation: Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). *CGMacros: a scientific dataset for personalized nutrition and diet monitoring* (version 1.0.0). PhysioNet. [doi:10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658).
- **Licensing:** data [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); project code MIT, separately. No raw redistribution; participant-level files and per-sample predictions remain local. Review dataset terms for derived materials; timestamps are time-shifted and recording gaps preclude equating calendar span with wear time.
- **Inputs:** static age, sex, BMI, HbA1c, fasting glucose, fasting insulin, and lipids; dynamic CGM, heart rate, activity, meals, and time-of-day features. Excluded predictors: subject identifiers, ethnicity, same-day fingersticks, and laboratory collection times. Sleep/HRV are absent; medication timing/dose and causal treatment response are not modeled. The carbohydrate-on-board proxy is heuristic.
- **Targets/design:** CGM glucose ≥180 mg/dL at +120 minutes, with additional 30- and 60-minute classification and future-glucose regression. `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` use nested participant-grouped tuning and held-out outer groups; primary classification repeats five distinct participant assignments. Inner selection uses AP for classification and MAE for regression; the bounded exploratory search still selects boundary settings.

## Evidence snapshot

Repeated 120-minute mean pooled PR-AUC (AP) **0.370** (range **0.358–0.382**), ROC-AUC **0.878** (**0.855–0.888**); equal-participant macro PR-AUC **0.243** (**0.241–0.246**), ROC-AUC **0.741** (**0.728–0.751**). Event rate: **6.26%**. Pooled scores weight timestamps and between-person ranking; repeated macro AP includes people with positive events, and macro ROC includes people with both classes.

**Same-held-out comparison, copied verbatim from [EVALUATION.md](EVALUATION.md):**

| Classifier / score | Pooled PR-AUC | Macro PR-AUC (45 people) | Pooled ROC-AUC | Macro ROC-AUC (29 people) |
|---|---:|---:|---:|---:|
| Full-fusion HGB | 0.382 | 0.157 | 0.888 | 0.751 |
| Sensor-only HGB | 0.376 | 0.126 | 0.876 | 0.676 |
| Five EHR variables + sensors | 0.400 | 0.152 | 0.885 | 0.740 |
| HbA1c + sensors | 0.357 | 0.148 | 0.869 | 0.717 |
| Persistence (current glucose) | 0.308 | 0.090 | 0.825 | 0.496 |
| Trend extrapolation | 0.226 | 0.100 | 0.777 | 0.523 |

This seed-42 comparison assigns zero AP to people without positive events, unlike repeated macro AP above. Reduced-EHR/HbA1c-only candidate selection is post-hoc and exploratory; earlier ablations use different folds and are not row-for-row comparators.

The demo cache uses the seed-42 multi-horizon regression folds from `StratifiedGroupKFold`, which differ from the five primary repeated-CV assignments behind the headline metrics, so the demo shows the same method but not row-for-row the headline folds.

**Personalization:** fixed first 72 hours for adaptation, later timestamps for testing; **43/45** classification-eligible by early availability, including 15 without positive test events. Adding early data to fitting lowers pooled regression MAE from **20.82 mg/dL** to **18.10 mg/dL** across all 45 (macro MAE: **21.07 mg/dL** to **18.17 mg/dL**). This is exploratory personalization, not a physiological twin or evidence of clinical benefit; score-only calibration preserves within-person ranking.

**Audit finding and resolution:** the chronological evaluation has been rerun using `adaptation_split.py`, purging 301 boundary-crossing labels (`t + 120 min <= cutoff`). The updated metrics above reflect this leakage-free evaluation. The dashboard withholds individual personalization comparisons until validated for interactive replay.

**Dashboard contract:** the forecast-only API omits future outcomes until explicit advancement to `t+120`. It displays aligned model-input snapshots, not raw sensor measurements or exact meal events. Primary seed-42 repeated-CV scores and optional seed-42 multi-horizon classifier/regressor outputs have separate source labels and must not be read as one experiment.

## Limitations and next gate

Calibration is descriptive for one cohort/assignment, not external validation. Subgroup participant-bootstrap intervals condition on fixed scores, omit refit uncertainty, and are especially unstable where only two healthy-range participants have events; prevalence differs by stratum. Alert cutoffs use training-only inner out-of-fold scores; the 30-minute episode merge and 60-minute cooldown are exploratory, ties can exceed nominal budgets, and participant-day denominators are approximate. The ≥40% training precision constraint failed on held-out windows (**37.2%**); window precision is distinct from episode workload. No external, prospective, fairness, or clinical-utility validation has been performed. The proposed locked external evaluation is specified in [External validation plan](EXTERNAL_VALIDATION_PLAN.md). Next gates are ethics/consent and secure governance, assessment of India's DPDP Act 2023, representative independent data, local calibration, prospective multi-site validation, and clinician-defined workload, actions, and escalation.

**Source:** [EVALUATION.md](EVALUATION.md), including metric conventions and full caveats. **Discussion:** [Jury Q&A](JURY_QA.md).
