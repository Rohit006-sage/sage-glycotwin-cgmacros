# Jury Q&A

## How much can we trust results from 45 US participants?

Trust them as retrospective proof-of-concept evidence, not as clinical validation: CGMacros contains 45 US participants, and the intended established-T2D review context is not equivalent to the mixed cohort. The diabetes-range HbA1c stratum has 14 people and is not a chart-confirmed T2D group. Participant-held-out evaluation reduces same-person leakage, but repeated assignments do not create an independent external cohort. Subgroup bootstrap intervals condition on fixed predictions and omit model-refit uncertainty; only two healthy-range participants have positive events, while calibration is descriptive for one cohort and assignment. No external, prospective, fairness, or clinical-utility validation has been performed, and the prototype excludes diagnosis, treatment, triage, live monitoring, and patient care.

## Why do pooled and participant-macro metrics differ?

Across five assignments, pooled PR-AUC is **0.370** (range **0.358–0.382**) and ROC-AUC **0.878** (**0.855–0.888**), versus macro PR-AUC **0.243** (**0.241–0.246**) and ROC-AUC **0.741** (**0.728–0.751**). Pooled metrics weight timestamps and reward between-person ranking, while macro summaries give each eligible person equal weight; AP must also be read against the **6.26%** event rate. Repeated macro AP averages participants with positive events, and macro ROC averages participants with both classes. The same-fold comparison instead assigns zero AP to people without events, so its macro AP is not directly interchangeable with the repeated table's convention.

## Did the alert-precision target fail on held-out people?

Yes: the earlier window-level policy maximizing recall at **≥40% training precision** achieved held-out window precision of **37.2%**, so the training constraint did not persist. Cutoffs were selected on inner out-of-fold training-participant predictions and applied unchanged to held-out people, not tuned on the test outcomes. The newer workload view uses an exploratory 30-minute episode merge and 60-minute cooldown; at 120 minutes the top-5% policy detects **122 / 384** episodes with **1.14** false alerts per approximate participant-day and **90 min** median lead. Episode detections per detections-plus-false-alerts are not window precision, and missing wear/gaps make the participant-day denominator approximate; score ties can also exceed nominal percentile budgets. Neither policy is a clinical alert rule: clinicians must define actions, grouping, suppression, acceptable workload, and prospective evaluation.

## Does adding EHR information actually help?

On the same seed-42 held-out folds, full-fusion HGB pooled PR-AUC is **0.382**, sensor-only **0.376**, and five EHR variables plus sensors **0.400**; the reduced variables are age, sex, BMI, HbA1c, and fasting glucose. Full versus reduced macro AP is **0.157** versus **0.152**, pooled ROC **0.888** versus **0.885**, and macro ROC **0.751** versus **0.740**, so no variant wins every summary. In the older fixed-fold analysis, removing baseline EHR yielded pooled/macro AP **0.350 / 0.198** and pooled/macro ROC **0.858 / 0.687**; those removals reused primary settings and were not separately retuned. That older analysis has different held-out participants and must not be compared row-for-row against the newer same-fold results. Reduced-EHR and HbA1c-only candidates were added after earlier results were seen, making candidate choice post-hoc, exploratory, and potentially optimistic rather than proof of incremental clinical utility.

The demo cache uses the seed-42 multi-horizon regression folds from `StratifiedGroupKFold`, which differ from the five primary repeated-CV assignments behind the headline metrics, so the demo shows the same method but not row-for-row the headline folds.

## Is this a real digital twin or just a predictor?

It is a twin-oriented forecasting and personalization prototype, not a physiological simulator or validated clinical digital twin. Static clinical context and dynamic sensor/meal features are combined with adaptation using the fixed first 72 hours, then evaluated on later timestamps; **43/45** qualify for classification by early availability, including 15 without positive test events. Adding early data to fitting reduces pooled regression MAE from **20.82 mg/dL** to **18.10 mg/dL** across all 45 participants (macro MAE: **21.07 mg/dL** to **18.17 mg/dL**). Score-only calibration preserves each person's ranking, so improved pooled metrics do not establish better within-person discrimination; none of these experiments estimates treatment response, causal mechanisms, or clinical benefit.

The evaluation was rerun using `adaptation_split.py`, purging 301 labels that crossed the 72-hour cutoff (`t + 120 min <= cutoff`). This confirms the improvement holds under a leakage-free protocol. Individual comparison in the dashboard remains withheld until validated for interactive replay.

## Why ≥180 mg/dL and a +120-minute horizon?

The event is CGM glucose **at least 180 mg/dL** at the selected future horizon, not any crossing during the intervening window. This is a domain-chosen upper-range boundary rather than a threshold tuned to the observed event rate, and the two-hour horizon is a project choice for retrospective clinician review rather than an established clinical alert interval. Additional 30- and 60-minute targets and future-glucose regression are evaluated to expose the trade-off among prediction, lead time, and workload. A useful clinical horizon would require a clinician-defined action and acceptable burden, not selection by PR-AUC alone; the forecast does not recommend treatment.

## Can this be used in India?

The workflow may motivate Indian research, but US-cohort risk levels, calibration, and performance cannot be assumed to transfer. Representative Indian data, local calibration, and prospective multi-site validation are required, with attention to subgroup uncertainty and intended established-T2D follow-up users. Any future study needs ethics/consent review, secure governance, assessment of India's Digital Personal Data Protection Act 2023, and clinician workflow, monitoring, and escalation design. CGMacros v1.0.0 data are licensed CC BY-NC-SA 4.0 separately from MIT project code; cite the dataset, review derived-material terms, and do not redistribute raw records or publish participant-level files or per-sample predictions. The current retrospective prototype must not connect to live devices, patient records, or care workflows.

## What would you do with more time?

First, obtain an independent, representative cohort and predefine the intended research use, comparison, and clinician-relevant outcomes before further candidate selection. Extend the bounded hyperparameter search because selected settings still hit boundaries, and evaluate personalization stability and calibration with independent evidence rather than assuming untested settings improve performance. Investigate timestamp-aligned sleep and HRV only if suitable data become available, while keeping the carbohydrate proxy explicitly heuristic and avoiding physiological or causal claims. Then pursue ethics-approved prospective multi-site evaluation of subgroup performance, clinician workload, and clinical utility, with secure governance and a defined escalation protocol before considering any care deployment.

Source for every reported metric and caveat: [EVALUATION.md](EVALUATION.md), especially repeated classification, same-held-out comparison, earlier ablations, alert workload, and chronological evaluation; dataset citation and separate license terms are also recorded there. See the compact [model card](MODEL_CARD.md).
