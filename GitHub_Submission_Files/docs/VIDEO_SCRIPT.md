# 20–25 minute submission video script

Jury discussion reference: [Jury Q&A](JURY_QA.md).

**Target runtime:** approximately 23–25 minutes. Read the sections aloud while showing the matching presentation slide or dashboard screen. Keep the visible banner that says this is a research prototype and retrospective replay.

## 0:00 — Introduction, team, and problem

“Hello, I am Manda Rohit Vardhan from team Sage at Amrita Vishwa Vidyapeetham, Hyderabad. This project is GlycoTwin, a proof-of-concept digital twin-oriented workflow for glucose forecasting. Our goal is not to claim a clinical product. Our goal is to demonstrate how static clinical context and dynamic wearable data can be fused to forecast a future glucose event for clinician review. The public repository and unlisted video link will be added before submission.”

“The healthcare use case is type 2 diabetes follow-up. ICMR-INDIAB estimated 101 million people in India had diabetes in 2021, with weighted prevalence of 11.4%. A clinician could benefit from a view that combines baseline context with recent glucose and lifestyle signals, but any Indian deployment would require local data, calibration, privacy review, and prospective validation.”

## 2:00 — Data, streams, constraints, and licensing

“The current proof of concept uses CGMacros version 1.0.0 from PhysioNet. It is a US cohort with 45 participants. The data include baseline demographics and laboratory measurements, two CGM streams, Fitbit activity, heart rate, and meal records. The repository does not redistribute the raw records. A clean reproduction downloads the dataset separately under the ignored data path.”

“The source dataset is licensed CC BY-NC-SA 4.0, while this project code is MIT licensed. Those are separate terms. We cite the dataset and keep raw data, detailed participant-level files, and per-sample predictions out of the public repository. Study timestamps are time-shifted, and a calendar span is not the same thing as continuous wear time.”

## 4:30 — Twin design

“The twin loop starts with a static EHR-like layer: age, sex, BMI, HbA1c, fasting glucose, fasting insulin, and lipid measurements. The dynamic layer contains time-aligned CGM, heart rate, activity, meal, and time-of-day signals. We join the baseline context to each forecast timestamp and calculate rolling glucose and time-above-range features, meal timing, a heuristic carbohydrate-on-board proxy, wearable features, and temporal features.”

“The personalization element is chronological per-person adaptation. The first 72 hours are used as a fixed adaptation or training window, subject to the documented eligibility rule. Later records are forecast after that window. This is a useful twin-oriented personalization experiment, but it is not a physiological simulator. The prototype does not simulate physiology, treatment response, or causal mechanisms.”

## 8:00 — Model and evaluation design

“The classifier and regressor are scikit-learn HistGradientBoosting models. The primary classification target is whether CGM glucose reaches 180 milligrams per decilitre at plus 120 minutes. Additional 30-minute and 60-minute horizons and regression are also evaluated. The two-hour horizon is a project choice for retrospective review, not a clinically established alert interval.”

“Evaluation keeps participant groups separated. We use nested participant-grouped tuning, with inner folds for model selection and outer held-out participants for evaluation. The repeated primary evaluation uses five distinct seeded participant assignments. Fold balancing considers participant count, sample count, and event count. Probability cutoffs are selected only from inner out-of-fold training-participant predictions and are treated as research comparisons, not deployment rules.”

## 12:00 — Results and honest interpretation

“For the 120-minute target, the dataset provides 45,447 usable examples, with 2,845 positive labels and 6.26% prevalence. Across five participant assignments, mean pooled PR-AUC is 0.370 with a range of 0.358 to 0.382, and pooled ROC-AUC is 0.878 with a range of 0.855 to 0.888. Equal-participant macro PR-AUC is 0.243 and macro ROC-AUC is 0.741.”

“These are not interchangeable summaries. Pooled metrics give every timestamp weight, while macro metrics give people more equal weight. PR-AUC depends on event prevalence. The subgroup intervals are based on participant bootstrap scores and do not include refitting uncertainty. Fifteen people have no positive test events in the personalization eligibility analysis. These results are exploratory and cannot be treated as Indian performance, external validation, or clinical benefit.”

“The same-fold comparison reports fused HGB pooled PR-AUC of 0.382, reduced five-variable EHR plus sensors at 0.400, persistence at 0.308, and trend extrapolation at 0.226. Candidate selection is post-hoc and exploratory. We show the comparison for transparency, not to claim that the reduced model is a final choice.”

## 15:00 — Personalization

“The chronological personalization result compares population-only forecasting with fitting on eligible people’s early records and evaluating later timestamps. Forty-three of 45 participants qualify under the documented early-availability rule. After purging 301 boundary-crossing labels, pooled and macro PR-AUC are 0.523 and 0.251 for the adapted setup versus 0.374 and 0.240 for population-only. Regression covers all 45 participants; pooled MAE falls from 20.82 to 18.10 milligrams per decilitre, and macro MAE from 21.07 to 18.17.”

“The correct interpretation is that early within-person data changed later forecasts in this retrospective experiment. It does not prove that an adaptation workflow is safe, stable, or useful in prospective care. It also does not turn the model into a physiological twin.”

“The adaptation split purges any early forecast row whose +120-minute outcome would fall after the 72-hour cutoff; 301 crossing labels were excluded in the rerun. The aggregate comparison uses this purged boundary. Individual paired personalization comparisons remain withheld from the dashboard.”

## 17:00 — Alert workload and episode view

“We also report an episode-level workload view. Under an exploratory 30-minute episode merge and 60-minute alert cooldown, top-five-percent training cutoffs detect 199 of 391 30-minute episodes, 197 of 389 60-minute episodes, and 122 of 384 120-minute episodes. False alerts range from 0.25 to 1.14 per approximate participant-day, with median lead times of 15, 30, and 90 minutes respectively.”

“These are retrospective workload summaries, not recommended alert settings. The merge and cooldown values need clinician input, prospective validation, and a defined follow-up action. Choosing a horizon requires a clinician and workflow context; an earlier warning is not automatically a better warning.”

## 18:30 — Live dashboard walkthrough

“I will now show the local replay dashboard. The banner states that this is a research-only retrospective replay. First, I point to the risk-ranked review list. The ranking is based on the held-out seed-42 replay score and is not triage. Next, I select an anonymized participant and move the historical time slider. The glucose trace is retrospective, and the observed future value is shown for evaluation review.”

“The forecast card shows the probability for glucose reaching 180 milligrams per decilitre at plus 120 minutes. The context cards show recent heart rate, activity, and meal information. The static EHR summary shows baseline HbA1c, BMI, and related fields. Nothing on this screen is a live device feed, a diagnosis, a treatment recommendation, or a clinical alert.”

“I first show the forecast-only state, then explicitly advance to the exact observed plus-120-minute outcome. The optional multi-horizon source pairs its own classifier score with its own regression estimate and uses different seed-42 StratifiedGroupKFold assignments from the primary repeated-CV headline results. The synchronized views show model-input snapshots and rolling meal summaries, not raw measurement streams.”

## 21:00 — Limitations, privacy, and deployment path

“The limitations are central. There are 45 participants, the cohort is from the United States, and performance varies by participant assignment. There is no external validation, prospective study, fairness assessment, or clinical utility study. Medication timing and dose are not modeled. Sleep and HRV are absent. The prototype is not a physiological simulator and must not be connected to live devices, patient records, or care workflows.”

“A future Indian deployment would need representative Indian data, consent and governance, local calibration, prospective multi-site validation, clinician workflow evaluation, monitoring, and an escalation protocol. The team would assess India's Digital Personal Data Protection Act 2023 and any applicable HIPAA obligations. The next technical steps include Indian data, sleep and HRV if suitable, secure deployment, and prospective validation.”

## 23:00 — Conclusion

“To conclude, GlycoTwin demonstrates a complete research workflow: static baseline context and dynamic streams are fused, a grouped forecasting model predicts a two-hour glucose event, chronological early-person adaptation is evaluated, alert workload is examined, and a clinician-oriented retrospective dashboard makes the result reviewable. The evidence is promising only as a proof of concept. It is not clinical validation and not a physiological simulation. Thank you.”

## Recording tips

- Record the PPTX with narration or use a screen recorder while showing the slides and local dashboard.
- If using YouTube, upload as **Unlisted**, copy the link into the README and submission form, then test it in a logged-out/private browser window.
- Keep the final recording between 22 and 25 minutes; pause briefly on the results tables and limitations.
- Never show the raw `data/` directory, participant-level CSVs, per-sample predictions, credentials, or local paths containing private information.
