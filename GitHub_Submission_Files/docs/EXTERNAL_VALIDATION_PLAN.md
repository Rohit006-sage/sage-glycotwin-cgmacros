# External validation plan

## Status

External validation has **not** been performed. The current evidence comes from 45 participants in the US CGMacros cohort. The published cross-validation results estimate participant-held-out performance within that cohort; they do not establish transportability to another site, population, device, or care setting.

## Data and governance gate

Before evaluation, identify an independent cohort with documented consent, ethics approval or exemption, data-use authority, and permission to create and share aggregate results. Record the site, recruitment period, population, devices, outcome definitions, missingness, and exclusions. Do not place identifiable or participant-level data in this repository.

The validation cohort must not influence feature selection, model selection, threshold selection, or the analysis plan. If access to an independent cohort is unavailable, report this gate as pending and do not imply external validation.

## Locked evaluation

Before accessing outcomes, freeze and record the code revision, runtime versions, feature schema, target definition, preprocessing, model configuration, and analysis plan. Evaluate the existing model without refitting or recalibrating on the external cohort. If local recalibration is later studied, report it as a separate secondary analysis with a clearly separated adaptation and evaluation period.

Report, at minimum:

- cohort and participant flow, with exclusions and missing-data coverage;
- event prevalence and participant-level event distribution;
- pooled and participant-macro average precision and ROC-AUC, with the averaging rules stated;
- calibration and Brier score, with uncertainty intervals clustered by participant;
- sensitivity, precision, and alert workload at the already specified operating points, without tuning them on external outcomes;
- regression MAE and RMSE at the exact target horizon;
- results by clinically justified subgroups only where sample sizes support useful uncertainty estimates.

Use participant-level confidence intervals. Report unstable or undefined subgroup estimates explicitly. Do not interpret statistical performance as clinical benefit or safety.

## Decision gate

An independent retrospective result can support transportability research, but it cannot establish clinical utility. Any prospective study requires local clinical governance, a pre-specified protocol, clinician-defined actions and workload limits, monitoring, and appropriate ethics and privacy review. The current prototype must remain disconnected from live care workflows.
