# GlycoTwin: CGMacros glucose-forecasting proof of concept

> **Research prototype only. Not clinically validated; do not use for diagnosis, treatment, triage, or patient care.**

## Results at a glance

- **Dataset:** all 45 CGMacros participants; 45,447 usable examples for the 120-minute target (2,845 positive labels; 6.26% prevalence). CGMacros is a US cohort, not an Indian cohort.
- **120-minute repeated grouped evaluation:** across five distinct participant assignments, mean pooled PR-AUC **0.370** (range 0.358–0.382) and ROC-AUC **0.878** (0.855–0.888); equal-participant macro PR-AUC **0.243** (0.241–0.246) and ROC-AUC **0.741** (0.728–0.751).
- **Chronological personalization:** first 72 hours are the fixed adaptation period with a purged boundary (301 boundary-crossing labels purged). Eligibility uses early availability only (≥200 labeled intervals), so **43/45** qualify; 15 have no positive test events. Adding early records to model fitting reaches pooled/macro PR-AUC **0.523 / 0.251**, versus **0.374 / 0.240** for population-only. Regression covers all 45; pooled MAE falls to **18.10 mg/dL** from **20.82 mg/dL** (macro MAE: **18.17 mg/dL** vs **21.07 mg/dL**).
- **Same-fold comparison:** on the same seed-42 held-out participants, fused HGB pooled PR-AUC is **0.382**, reduced five-variable EHR + sensors **0.400**, persistence **0.308**, and trend extrapolation **0.226**. Post-hoc candidate choice is exploratory.
- **Episode workload:** under an exploratory 30-minute episode merge and 60-minute alert cooldown, top-5% training cutoffs detect **199/391** 30-minute episodes, **197/389** 60-minute episodes, and **122/384** 120-minute episodes. False alerts range from **0.25 to 1.14 per approximate participant-day**, with median lead times of **15, 30, and 90 minutes**, respectively. These rules are not clinical standards.
- **Artifacts:** [aggregate metrics](results/aggregate_metrics.json), [subgroup bootstrap intervals](results/subgroup_bootstrap_metrics.json), [preserved earlier results](results/previous_aggregate_results.json), [calibration plot](figures/calibration_120m.svg), [repeated-CV spread](figures/repeated_cv_spread.svg), [horizon comparison](figures/multi_horizon.svg), [same-fold model comparison](figures/same_fold_model_comparison.svg), [episode workload by horizon](figures/episode_alert_horizon_comparison.svg), [chronological comparison](figures/within_person_comparison.svg), and [submission checklist](SUBMISSION_CHECKLIST.txt).

## Problem and intended use

Type 2 diabetes (T2D) is an important Indian health priority. ICMR-INDIAB estimated 101 million people in India had diabetes in 2021, with weighted prevalence of 11.4%. This project asks whether baseline clinical context plus recent continuous glucose monitoring (CGM), heart rate, activity, and meal information can forecast elevated glucose. The **two-hour** horizon gives a prototype follow-up window for a clinician to review; it is a project choice, not a clinically established alert interval. The event is CGM glucose **at least 180 mg/dL** at the selected future horizon, a domain-chosen upper-range boundary rather than a threshold tuned to the observed event rate. [ICMR-INDIAB study](https://doi.org/10.1016/S2213-8587(23)00119-5) · [ADA Standards of Care 2026, Section 6](https://diabetesjournals.org/care/article/49/Supplement_1/S132/163927/6-Glycemic-Goals-Hypoglycemia-and-Hyperglycemic)

**Intended user:** a diabetologist or endocrinologist reviewing a person with established T2D during follow-up. In the concept workflow, the clinician sees a risk-ranked patient list, opens a recent glucose trace and two-hour forecast, reviews meal and activity context, and decides whether to verify the data or follow their usual care process. The prototype does not prescribe medication or treatment.

The approach transfers as a **data-fusion and forecasting workflow**, not as validated Indian risk estimates. CGMacros is a small US study with 45 participants, including 14 in the diabetes-range HbA1c stratum. Its learned risk levels, calibration, and performance cannot be assumed to transfer to India; representative local data and prospective validation are required.

## Data and twin scope

The source is [CGMacros v1.0.0 on PhysioNet](https://doi.org/10.13026/3z8q-x658), a multimodal study with baseline demographics and blood measurements, two CGM streams, Fitbit activity, heart rate, and meal records. The local participant timestamp spans range from about 7.2 to 19.2 calendar days (median about 10.6); the average is about 10 days, and the longest spans include recording gaps. Calendar span is not continuous wear time, and study dates are time-shifted. Raw source data are not included in this repository. Cite the dataset in public materials and review its terms before sharing derived work.

The model fuses static baseline age, sex, BMI, HbA1c, fasting glucose, fasting insulin, and lipid measurements with time-indexed CGM, heart-rate, activity, meal, and time-of-day features. Subject identifiers, ethnicity, same-day fingersticks, and laboratory collection times are excluded as predictors. HbA1c strata are analytic categories, not chart diagnoses.

This is a **twin-oriented forecasting prototype**, not a physiological simulator or a validated clinical digital twin. It now compares population forecasting with early-person adaptation and person-only fitting, but it does not simulate physiology, treatment response, or causal mechanisms. Sleep and HRV are not available in the streams used here; they could be added later as timestamp-aligned inputs if suitable data become available.

## Architecture and presentation

- [Architecture diagram (PDF)](architecture.pdf), [architecture presentation (PPTX)](architecture.pptx), and [editable Mermaid source](architecture.mmd): `bio.csv` + CGM + heart rate + activity + meals → feature fusion → model → forecast → dashboard.
- [Project presentation (PPTX)](GlycoTwin_CGMacros_Prototype.pptx), synchronized with the current repeated evaluation, calibration, cohort caveats, and personalization design.
- The local replay dashboard is a retrospective research view, not a live monitor or clinical system. Annotated concept screens are included in the presentation.

## Evaluation design and model

All available participants and eligible labeled samples are used. At 15-minute replay points, the binary outcome is whether the future CGM value reaches 180 mg/dL. The primary forecast is 120 minutes; additional targets are 30 and 60 minutes. Regression predicts the future glucose value at the same horizon.

The classifier and regressor are scikit-learn `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` models. **There are no epochs:** `max_iter` is a boosting-iteration parameter. The nested search tests four representative configurations spanning learning rates 0.02–0.10, `max_iter` 300–1,000, `max_leaf_nodes` 3–31, `min_samples_leaf` 50–1,000, and L2 regularization 0–10. Classifier settings are selected by inner participant-grouped average precision; regression settings use inner participant-grouped MAE. Outer participant groups remain held out. Early stopping is disabled; iteration count is chosen among the tested values using only inner training-participant folds.

Repeated primary evaluation uses five distinct seeded participant assignments, with folds balanced on participant count, sample count, and event count. The inner search remains grouped by participant. Because edge values were still selected in many fits, the search is a bounded exploratory grid and should be extended only with compute and an independent validation cohort; the edge counts are reported in the aggregate JSON.

| Search boundary selected | Outer fits (of 25) |
|---|---:|
| `learning_rate` minimum 0.02 / maximum 0.10 | 14 / 1 |
| `max_iter` maximum 1,000 | 15 |
| `max_leaf_nodes` minimum 3 / maximum 31 | 14 / 1 |
| `min_samples_leaf` minimum 50 / maximum 1,000 | 14 / 9 |
| `l2_regularization` minimum 0 / maximum 10 | 22 / 1 |

Selections still land on several search boundaries, particularly the minimum learning rate, minimum leaf count, minimum leaf size, zero L2, and maximum iteration count. These counts mean the best of the tested representative configurations often sat at a boundary; they do not show that untested values beyond the stated range would improve performance.

At each outer fold, probability cutoffs are selected from inner out-of-fold predictions for training participants only, then applied unchanged to held-out participants. We evaluate top 1%, 5%, and 10% training-score budgets and a cutoff with maximum inner-OOF recall at ≥40% training precision. Cutoffs are for research comparisons, not deployment.

### Repeated 120-minute classification results

Pooled values treat every time point as an observation. Macro PR-AUC averages participant AP over people with at least one positive event; macro ROC-AUC averages people with both classes, giving each person equal weight. PR-AUC depends on event prevalence; the 120-minute cohort prevalence is 6.26%.

| Fold-assignment seed | Pooled PR-AUC | Macro PR-AUC | Pooled ROC-AUC | Macro ROC-AUC |
|---:|---:|---:|---:|---:|
| 11 | 0.358 | 0.242 | 0.882 | 0.737 |
| 23 | 0.364 | 0.241 | 0.881 | 0.743 |
| 42 | 0.382 | 0.244 | 0.888 | 0.751 |
| 73 | 0.371 | 0.244 | 0.855 | 0.728 |
| 101 | 0.375 | 0.246 | 0.885 | 0.745 |
| **Mean (SD)** | **0.370 (0.009)** | **0.243 (0.002)** | **0.878 (0.013)** | **0.741 (0.009)** |

The earlier grouped evaluation reported pooled ROC-AUC **0.878** and participant-macro ROC-AUC **0.749**. Pooled ROC-AUC is higher because pooled scores also reward ranking between people; participant-macro ROC-AUC measures ranking within each person. The current repeated evaluation shows the same distinction. Macro PR-AUC in this table averages participants with at least one positive; macro ROC-AUC averages the 29 participants with both classes. Neither summary substitutes for external validation.

### HbA1c strata and participant-cluster uncertainty

These subgroup metrics use seed-42 held-out predictions from the current feature set. The constant-prevalence PR-AUC baseline is the event rate, which differs substantially across groups. Confidence intervals resample whole participants 2,000 times and condition on the fixed OOF scores; they do not include model-refit uncertainty.

| Baseline HbA1c stratum | People | Events / people with ≥1 event | Event rate / constant-AP baseline | Pooled PR-AUC (95% participant-bootstrap CI) | Pooled ROC-AUC (95% CI) | Macro PR-AUC | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Healthy-range <5.7% | 15 | 106 / 2 | 0.7% | 0.044 (0.000–0.264) | 0.893 (0.343–0.944) | 0.203 (n=2) | 0.642 (n=2) |
| Prediabetes-range 5.7–<6.5% | 16 | 439 / 13 | 2.8% | 0.164 (0.037–0.274) | 0.832 (0.739–0.891) | 0.100 (n=13) | 0.713 (n=13) |
| Diabetes-range ≥6.5% | 14 | 2,300 / 14 | 15.9% | 0.430 (0.266–0.586) | 0.803 (0.738–0.851) | 0.382 (n=14) | 0.802 (n=14) |

Only **two healthy-range participants** contributed any positive events, so the wide intervals and low macro sample count are central to interpreting that result; only 1,781 of 2,000 healthy-range bootstrap draws contained events. The diabetes-range stratum is the closest available subgroup to the intended follow-up context, but it is an HbA1c band, not a chart-confirmed T2D diagnosis. Its ROC-AUC (**0.803**) is lower than the pooled-cohort value (**0.888**). The pooled AP/ROC and macro metrics answer different questions; do not compare PR-AUC across strata without their different prevalence baselines. The all-participant participant-bootstrap intervals are pooled PR-AUC **0.216–0.529** and ROC-AUC **0.825–0.931**.

### Same-held-out model and baseline comparison

All models below use the same five outer participant folds (seed 42). Candidate HGB classifiers are tuned by inner participant-grouped AP only; the primary full-fusion scores are reused from the corresponding held-out fold. Persistence and trend use no learned classifier. For this comparison, equal-participant macro PR-AUC assigns 0 to a participant with no positive events; macro ROC-AUC is defined only for participants with both classes (29 people). This is a different AP convention from the repeated-results table above, where AP is averaged only where defined.

| Classifier / score | Pooled PR-AUC | Macro PR-AUC (45 people) | Pooled ROC-AUC | Macro ROC-AUC (29 people) |
|---|---:|---:|---:|---:|
| Full-fusion HGB | 0.382 | 0.157 | 0.888 | 0.751 |
| Sensor-only HGB | 0.376 | 0.126 | 0.876 | 0.676 |
| Five EHR variables + sensors | 0.400 | 0.152 | 0.885 | 0.740 |
| HbA1c + sensors | 0.357 | 0.148 | 0.869 | 0.717 |
| Persistence (current glucose) | 0.308 | 0.090 | 0.825 | 0.496 |
| Trend extrapolation | 0.226 | 0.100 | 0.777 | 0.523 |

The reduced-EHR and HbA1c-only variants were added after earlier results had been seen. Choosing among these results is exploratory and may be optimistic. Full-fusion HGB regression on the same held-out rows had pooled MAE/RMSE **20.55/29.46 mg/dL** and macro **20.73/27.11**; persistence was **24.10/36.94** pooled and **23.98/33.71** macro, while trend extrapolation was **45.30/69.22** pooled and **45.08/64.99** macro.

### Episode-level alert workload

These event summaries replace point-alert counts as the workload view. For each horizon, future CGM readings ≥180 mg/dL are grouped into one episode when consecutive positive readings are no more than 30 minutes apart. A candidate alert is emitted at the first cutoff-crossing prediction, followed by a 60-minute cooldown. Both values are exploratory analysis choices, not clinical conventions; clinicians must define appropriate event grouping, suppression, action window, and workload before any prospective study. The cutoff comes from inner out-of-fold predictions of training participants only and is applied unchanged to held-out participants. The top-percentile cutoffs can exceed the nominal alert budget when scores tie. The 40%-precision cutoff maximizes training inner-OOF recall subject to ≥40% training point precision; held-out event workload does not preserve that target.

In the table, an episode is detected early only when an alert occurs before its onset and its forecast target falls within the episode. Alerts after onset are counted separately. False-alert rates divide false cooldown-grouped alerts by prediction opportunities/96; missing wear and recording gaps mean this is an approximate participant-day denominator. “Detected people” counts participants with at least one episode detected; “repeat/no alert” count people with ≥2 cooldown-grouped alerts / no emitted alert.

| Horizon | Training-only cutoff | Episodes detected / total | Episode recall | Episode detections per (detections + false alerts) | False alerts / approximate participant-day | Median lead | Detected people | Repeat / no alert people |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30 min | Top 1% | 3 / 391 | 0.8% | 100.0% | 0.00 | 30 min | 3 | 9 / 35 |
| 30 min | Top 5% | 199 / 391 | 50.9% | 63.0% | 0.25 | 15 min | 23 | 26 / 19 |
| 30 min | Top 10% | 265 / 391 | 67.8% | 23.9% | 1.78 | 30 min | 25 | 32 / 10 |
| 30 min | ≥40% training precision | 234 / 391 | 59.8% | 12.4% | 3.48 | 30 min | 28 | 39 / 3 |
| 60 min | Top 1% | 27 / 389 | 6.9% | 69.2% | 0.03 | 15 min | 7 | 8 / 35 |
| 60 min | Top 5% | 197 / 389 | 50.6% | 39.5% | 0.64 | 30 min | 17 | 18 / 26 |
| 60 min | Top 10% | 267 / 389 | 68.6% | 21.3% | 2.08 | 30 min | 21 | 25 / 20 |
| 60 min | ≥40% training precision | 273 / 389 | 70.2% | 16.9% | 2.83 | 45 min | 21 | 25 / 19 |
| 120 min | Top 1% | 59 / 384 | 15.4% | 39.6% | 0.19 | 90 min | 9 | 9 / 36 |
| 120 min | Top 5% | 122 / 384 | 31.8% | 18.4% | 1.14 | 90 min | 13 | 14 / 30 |
| 120 min | Top 10% | 189 / 384 | 49.2% | 14.5% | 2.36 | 90 min | 18 | 21 / 24 |
| 120 min | ≥40% training precision | 144 / 384 | 37.5% | 15.5% | 1.66 | 90 min | 14 | 16 / 28 |

This comparison shows the trade-off among detection, lead time, and workload; it does not select a clinical horizon. The 30-minute horizon has much stronger ranking metrics, while 120 minutes can provide more lead time when an event is detected. Horizon choice must reflect a clinician-defined action and acceptable burden, not PR-AUC alone. Post-onset alerts targeting an already-started episode are reported separately in the aggregate JSON. See [episode workload figure](figures/episode_alert_horizon_comparison.svg) and [aggregate event metrics](results/aggregate_metrics.json).

For continuity, the earlier **window-level** 120-minute operating-point results are also retained below. A positive 15-minute prediction window was counted separately; these values do not merge events or apply a cooldown and therefore should not be interpreted as patient alert episodes.

| Training policy | Held-out window precision | Held-out window recall | Positive windows / approximate participant-day | Participants with ≥1 positive window caught |
|---|---:|---:|---:|---:|
| Top 1% | 58.6% | 11.0% | 1.13 | 9.6 |
| Top 5% | 45.2% | 40.3% | 5.37 | 12.2 |
| Top 10% | 33.7% | 59.5% | 10.62 | 17.6 |
| Max recall at ≥40% training precision | 37.2% | 48.8% | 7.90 | 13.8 |

The training precision constraint did **not** persist on held-out people. The event-level analysis above applies cooldown and groups observed glucose episodes, and is the more appropriate exploratory workload summary.

The [calibration plot](figures/calibration_120m.svg) summarizes deciles from held-out predictions for assignment seed 42. It is descriptive, based on this one cohort and assignment, and is not external calibration validation.

### Multi-horizon forecasts and regression

These nested grouped metrics use the 45-participant cohort. The 30/60/120-minute rows show pooled and equal-participant macro classification and regression metrics together. Macro PR-AUC includes participants with at least one positive event; macro ROC-AUC includes participants with both classes. Regression includes everyone.

| Horizon | Pooled PR-AUC | Macro PR-AUC | Pooled ROC-AUC | Macro ROC-AUC | Pooled MAE / RMSE (mg/dL) | Macro MAE / RMSE (mg/dL) |
|---:|---:|---:|---:|---:|---:|---:|
| 30 min | 0.901 | 0.668 | 0.989 | 0.979 | 8.59 / 12.49 | 8.60 / 12.15 |
| 60 min | 0.687 | 0.382 | 0.944 | 0.854 | 15.15 / 22.01 | 15.17 / 20.83 |
| 120 min | 0.396 | 0.222 | 0.855 | 0.730 | 20.53 / 29.68 | 20.66 / 27.26 |

The 120-minute classification row above is the single nested horizon-comparison run; the repeated estimates are shown separately in the preceding table. The [horizon plot](figures/multi_horizon.svg) shows pooled and macro PR-AUC side by side.

### Feature-set comparisons and ablations

The following feature-variant and ablation table preserves an earlier single fixed-fold analysis for traceability; it does not use the same held-out participants as the newer comparison above and should not be compared row-for-row. The reduced-EHR and HbA1c-only candidates were added after earlier results had been observed, so choosing among them is exploratory and may be optimistic.

| Feature set / model | Pooled PR-AUC | Macro PR-AUC | Pooled ROC-AUC | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| Full baseline EHR + sensors, primary nested fit | 0.396 | 0.222 | 0.855 | 0.730 |
| Five baseline variables (age, sex, BMI, HbA1c, fasting glucose) + sensors | 0.396 | 0.233 | 0.885 | 0.733 |
| HbA1c only + sensors | 0.380 | 0.228 | 0.871 | 0.718 |

The following ablations remove each feature group from the full 120-minute feature set. They reuse the primary outer-fold settings and are **not separately retuned**, so interpret them as diagnostic comparisons rather than optimized alternatives.

| Removed feature group | Pooled PR-AUC | Macro PR-AUC | Pooled ROC-AUC | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| Glucose history and rolling statistics | 0.332 | 0.220 | 0.851 | 0.712 |
| Time above 140/180 mg/dL | 0.399 | 0.224 | 0.857 | 0.733 |
| Wearable and activity | 0.407 | 0.219 | 0.860 | 0.732 |
| Meals and carbohydrate-on-board proxy | 0.377 | 0.189 | 0.853 | 0.700 |
| Baseline EHR | 0.350 | 0.198 | 0.858 | 0.687 |
| Time of day | 0.350 | 0.200 | 0.852 | 0.682 |

Added features include minutes above 140 and 180 in the prior hour, minutes of glucose observed, meal timing, a **heuristic** carbohydrate-on-board feature with linear decay over three hours, and rolling glucose mean, standard deviation, minimum, and maximum over 30, 60, and 120 minutes. The carbohydrate proxy is not a validated absorption or physiological model.

### Chronological within-person evaluation

The primary chronological split uses a fixed **first 72 hours** for adaptation/training and all later timestamps for testing. This fixed early window avoids choosing adaptation duration from each participant's future recording length. Primary classification eligibility requires at least 200 usable labeled intervals in the early window and one or more later test intervals; it does **not** depend on future event counts. **43 of 45** participants qualify, including 15 with no positive test windows. PR-AUC is averaged over the 28 eligible participants with at least one positive test window; ROC-AUC is averaged over the 28 eligible participants with both test classes. Regression includes all 45 participants. For each participant, the population hyperparameters are taken from grouped CV where that person was held out.

| Strategy | Eligible people (classification) | Pooled PR-AUC | Macro PR-AUC | Pooled ROC-AUC | Macro ROC-AUC | Pooled MAE / RMSE (mg/dL), all 45 | Macro MAE / RMSE (mg/dL), all 45 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Population-only | 43 | 0.374 | 0.240 | 0.881 | 0.736 | 20.82 / 29.98 | 21.07 / 27.39 |
| Population + early-window calibration | 43 | 0.538 | 0.240 | 0.927 | 0.736 | Score-only; no glucose-regression change | Score-only; no glucose-regression change |
| Population + early data added to model fit | 43 | 0.523 | 0.251 | 0.929 | 0.759 | 18.10 / 26.44 | 18.17 / 24.45 |
| Person-only, early segment | 43 | 0.340 | 0.140 | 0.867 | 0.546 | 20.54 / 29.59 | 20.55 / 27.04 |

The regularized one-dimensional calibration layer is fit only to the participant's first-72-hour model scores when that early segment has both classes and at least five positive windows (20 participants); otherwise it falls back to the population score. This fallback is not an eligibility filter. The calibration layer leaves within-person AP and ROC unchanged because it preserves each participant's score order; higher pooled metrics reflect changes to between-person score ordering, not better within-person discrimination. It reduces pooled Brier from 0.050 to 0.043. These strategies are exploratory; none models an intervention or physiological treatment response. The outcome-selected event-rich subgroup (16 participants, ≥10 positive windows in both segments) is descriptive only and is not used for primary claims. This evaluation does not establish clinical benefit.

**Audit finding and resolution:** `within_person_validation.py` previously split by forecast timestamp (`t < adaptation_end`) rather than label-availability timestamp, which allowed 7 labels per participant to cross the 72-hour boundary. The evaluation has now been rerun using `adaptation_split.py`, purging 301 boundary-crossing labels (`t + 120 min <= cutoff`). The updated metrics above reflect this leakage-free evaluation. Until the UI display is verified, the dashboard withholds individual personalization comparisons.

### Preserved earlier results

Earlier runs are preserved for traceability; their feature matrices, grouping assignments, and evaluation settings differ, so do not read the rows as a single monotonic benchmark. Full aggregate snapshots are in [previous results JSON](results/previous_aggregate_results.json).

| Historical evaluation | Pooled PR-AUC | Participant-macro PR-AUC | Pooled ROC-AUC | Participant-macro ROC-AUC |
|---|---:|---:|---:|---:|
| Original baseline run | 0.401 | Not reported | 0.883 | Not reported |
| First documented baseline run | 0.401 | Not reported | 0.882 | Not reported |
| Earlier nested grouped evaluation | 0.388 | 0.252 | 0.878 | 0.749 |

The small change between the two baseline snapshots followed a 30-minute maximum age for carried-forward sensor values, which changed predictor values near recording gaps; the sample labels and participant folds did not change. The archived snapshot contains the HbA1c strata and the earlier model comparisons. In those earlier strata, event prevalence ranged from 0.7% to 15.9%; PR-AUC must be read against each stratum's prevalence baseline. The diabetes-range stratum (14 people) is closer to the intended follow-up context than the mixed-cohort headline, but it is not a confirmed T2D diagnosis group.

## Results files and regeneration

The `docs/results/` JSON and `docs/figures/` SVG files contain aggregate metrics and plots only. No per-sample derived predictions or patient-level feature rows are published. Detailed held-out predictions are created locally under the git-ignored `data/processed/revised_full/` directory when the analysis is regenerated.

External validation has not been performed. A locked evaluation plan and data-governance prerequisites are documented in [EXTERNAL_VALIDATION_PLAN.md](EXTERNAL_VALIDATION_PLAN.md).

## Run

Requires Python 3.11+ and the packages in [requirements.txt](../requirements.txt). Obtain CGMacros v1.0.0 separately from [PhysioNet](https://doi.org/10.13026/3z8q-x658) and place the extracted source files under `data/raw/cgmacros/`. Raw data are not included here. On the current Windows machine, the expanded repeated grouped evaluation took about 45 minutes; same-fold comparison about 4 minutes; episode alert analysis about 3 minutes; fixed-window personalization about 3 minutes; and 2,000 participant-cluster bootstrap draws about 2 minutes. **Plan for roughly 57 minutes total**, not including environment setup or data download. Runtime varies with hardware and package environment.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\run_all.py
.\.venv\Scripts\python.exe .\revised_evaluation.py
.\.venv\Scripts\python.exe .\fair_baseline_comparison.py
.\.venv\Scripts\python.exe .\alert_episode_evaluation.py
.\.venv\Scripts\python.exe .\within_person_validation.py
.\.venv\Scripts\python.exe .\subgroup_bootstrap_summary.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
python revised_evaluation.py
python fair_baseline_comparison.py
python alert_episode_evaluation.py
python within_person_validation.py
python subgroup_bootstrap_summary.py
```

The evaluation scripts write aggregate-only report files under `docs/results/` and `docs/figures/`. Detailed out-of-fold predictions remain local and git-ignored. Do not publish per-sample prediction CSVs, participant-level result files, or any other per-sample derived data; the run commands above regenerate those files locally.

To replay the dashboard separately after training:

```powershell
.\.venv\Scripts\python.exe .\serve_cgmacros_dashboard.py
```

Open <http://127.0.0.1:8765>. It binds to loopback and displays retrospective held-out forecasts, not live monitoring.

The dashboard's primary source is the seed-42 primary repeated-CV classifier replay. Its optional source is the paired classifier/regressor output from the separate seed-42 multi-horizon `StratifiedGroupKFold` run; source labels distinguish these designs. The API returns only metadata at `/api/data`; `/api/replay` keeps outcomes hidden until explicit 120-minute advancement. It uses the saved exact target, never a nearest-observation substitution, and exposes missing states instead of inventing measurements. Timeline values are model-input snapshots on eligible replay timestamps, including rolling two-hour meal/carbohydrate summaries, not an uninterrupted raw stream.

## Limitations and deployment path

- Only 45 participants are available, and results vary across participant assignments. The pooled, participant-macro, and chronological estimates answer different questions.
- The data are from the United States. India-specific model development requires representative Indian data, local calibration, and prospective multi-site validation.
- This is not a physiological simulator and does not estimate causal treatment response; medication timing and dose are not modeled. Sleep and HRV are absent from the current streams.
- Episode merge and alert cooldown values in the retrospective workload analysis are exploratory. They need clinician input, prospective validation, and a defined follow-up action before any alert workflow is considered.
- No external validation, prospective study, fairness assessment, or clinical utility study has been performed. Do not connect this prototype to live devices, patient records, or care workflows.
- A deployment path requires ethics and privacy review, secure data governance, prospective validation, clinician workflow evaluation, monitoring, and a clear escalation protocol. Assess India's [Digital Personal Data Protection Act 2023](https://www.meity.gov.in/static/uploads/2024/02/Digital-Personal-Data-Protection-Act-2023.pdf). HIPAA applies only where the responsible entities and data fall within its scope; see [HHS covered entities and business associates](https://www.hhs.gov/hipaa/for-professionals/covered-entities/index.html).

## Team, submission, and code license

- **Team:** Sage
- **Team leader:** Manda Rohit Vardhan
- **Members and roles:** Manda Rohit Vardhan — Project Lead and Developer
- **College:** Amrita Vishwa Vidyapeetham, Hyderabad
- **Incubator:** None
- **Recommended repository name:** `sage-glycotwin-cgmacros`
- **Demo video:** To be added after uploading the unlisted YouTube video (15–20 minutes)
- **Public repository:** [https://github.com/Rohit006-sage/sage-glycotwin-cgmacros](https://github.com/Rohit006-sage/sage-glycotwin-cgmacros)

Project code is licensed under the [MIT License](../LICENSE). This code license is separate from the CGMacros dataset license, which is [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Review dataset terms before distributing data or derived materials.

## Dataset citation

Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). *CGMacros: a scientific dataset for personalized nutrition and diet monitoring* (version 1.0.0). PhysioNet. [doi:10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658).
