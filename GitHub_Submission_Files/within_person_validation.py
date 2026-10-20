from __future__ import annotations

"""Chronological within-participant adaptation evaluation for the 120-minute task."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from adaptation_split import split_adaptation
from revised_evaluation import ROOT, OUT, DOC_RESULTS, DOC_FIGURES, make_clf, make_reg, class_metrics, participant_macro, regression_metrics, save_svg

PRED_PATH = OUT / "rich_120m_samples_local.csv"
AGG_PATH = DOC_RESULTS / "aggregate_metrics.json"
ADAPTATION_HOURS = 72
LABEL_HORIZON_MINUTES = 120  # a row's label is observed at timestamp + this horizon
MIN_EARLY_INTERVALS = 200


def mode_params(folds, key):
    from collections import Counter
    values = [json.dumps(f[key], sort_keys=True) for f in folds]
    return json.loads(Counter(values).most_common(1)[0][0])


def predict_classifier(train_x, train_y, test_x, params):
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(train_y.mean()))
    model = make_clf(params)
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def predict_regressor(train_x, train_y, test_x, params):
    model = make_reg(params)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def macro_regression(frame, pred_col):
    grouped=frame.assign(abs_error=(frame[pred_col]-frame.target_glucose).abs(),squared_error=(frame[pred_col]-frame.target_glucose)**2).groupby("participant_id")
    return {"mae_mg_dl":float(grouped.abs_error.mean().mean()),
            "rmse_mg_dl":float(np.sqrt(grouped.squared_error.mean()).mean())}


def main():
    frame=pd.read_csv(PRED_PATH,dtype={"participant_id":str},parse_dates=["timestamp"])
    metrics=json.loads((OUT/"revised_full_metrics.json").read_text(encoding="utf-8"))
    features=metrics["dataset"]["features"]
    class_nested=metrics["primary_120m_repeated_cv"]["seed_results"]
    class_fold_params={f["fold"]:f["selected_params"] for f in next(x for x in class_nested if x["seed"]==42)["folds"]}
    regression_folds=metrics["multi_horizon"]["120"]["regression_folds"]
    ycol="label";target="target_glucose"
    frame=frame.sort_values(["participant_id","timestamp"]).reset_index(drop=True)
    participant_results=[]; prediction_rows=[]
    calibration_fit_participants=0
    for pid, part in frame.groupby("participant_id",sort=True):
        part=part.reset_index(drop=True)
        # Purged boundary: a training row's t+120 label must already be observed at the
        # adaptation cutoff. Rows forecast before the cutoff whose label arrives after it
        # are dropped from both train and test, so no outcome beyond the cutoff is used.
        split=split_adaptation(part.timestamp,ADAPTATION_HOURS,LABEL_HORIZON_MINUTES)
        train=part.loc[split.train].copy()
        test=part.loc[split.test].copy()
        purged_samples=int(split.purged.sum())
        # Primary eligibility uses only pre-test data availability, never held-out outcomes.
        qualifies=int(len(train)>=MIN_EARLY_INTERVALS and len(test)>0)
        # Select a hyperparameter set from the grouped-CV outer fold in which
        # this participant was entirely held out. Thus the person's future
        # segment did not take part in hyperparameter selection.
        fold=next(f["fold"] for f in next(x for x in class_nested if x["seed"]==42)["folds"]
                  if str(pid) in f.get("test_participant_ids", []))
        cparams=class_fold_params[fold]
        # Classification repeated-CV folds and the single nested regression
        # split may use different seeded assignments; match this person by ID.
        rparams=next(f["regression_params"] for f in regression_folds
                     if str(pid) in f.get("test_participant_ids", []))
        xtrain=train[features].astype(float); xtest=test[features].astype(float)
        ytrain=train[ycol].to_numpy(int); ytest=test[ycol].to_numpy(int)
        outcomes={}
        # Population-only: all records for the other 44 people.
        population_train=frame.loc[frame.participant_id.ne(pid)]
        Xpop=population_train[features].astype(float); ypop=population_train.label.to_numpy(int)
        population_model=make_clf(cparams).fit(Xpop,ypop)
        population_train_score=population_model.predict_proba(xtrain)[:,1]
        population_test_score=population_model.predict_proba(xtest)[:,1]
        outcomes["population_only"]=population_test_score
        # Small person-specific Platt layer on the fixed, label-purged early window only.
        # When the 72-hour data do not contain at least five positive windows
        # and both classes, keep the population score unchanged.
        calibration_fitted=int(ytrain.sum()>=5 and np.unique(ytrain).size==2)
        if calibration_fitted:
            eps=1e-5
            early_logit=np.log(np.clip(population_train_score,eps,1-eps)/(1-np.clip(population_train_score,eps,1-eps))).reshape(-1,1)
            test_logit=np.log(np.clip(population_test_score,eps,1-eps)/(1-np.clip(population_test_score,eps,1-eps))).reshape(-1,1)
            calibrator=LogisticRegression(C=0.1,solver="lbfgs",max_iter=500,random_state=42)
            calibrator.fit(early_logit,ytrain)
            outcomes["population_plus_early_calibration"]=calibrator.predict_proba(test_logit)[:,1]
            calibration_fit_participants+=1
        else:
            outcomes["population_plus_early_calibration"]=population_test_score
        # Population + early adaptation: add only this participant's label-available
        # early rows (purged at the 72-hour cutoff) to the population training set.
        adapt_x=pd.concat([Xpop,xtrain],ignore_index=True)
        adapt_y=np.concatenate([ypop,ytrain])
        outcomes["population_plus_early_adaptation"]=predict_classifier(adapt_x,adapt_y,xtest,cparams)
        # Per-person-only: fit solely on this person's early segment.
        outcomes["per_person_only"]=predict_classifier(xtrain,ytrain,xtest,cparams)
        reg_outcomes={}
        reg_ytrain=train[target].to_numpy(float); reg_ytest=test[target].to_numpy(float)
        reg_outcomes["population_only"]=predict_regressor(Xpop,population_train[target].to_numpy(float),xtest,rparams)
        # Classification-only score calibration does not alter glucose regression.
        reg_outcomes["population_plus_early_calibration"]=reg_outcomes["population_only"]
        adapt_reg_x=pd.concat([Xpop,xtrain],ignore_index=True)
        adapt_reg_y=np.concatenate([population_train[target].to_numpy(float),reg_ytrain])
        reg_outcomes["population_plus_early_adaptation"]=predict_regressor(adapt_reg_x,adapt_reg_y,xtest,rparams)
        reg_outcomes["per_person_only"]=predict_regressor(xtrain,reg_ytrain,xtest,rparams)
        event_rich=int(ytrain.sum()>=10 and ytest.sum()>=10)
        participant_results.append({"participant_id":str(pid),"train_samples":len(train),"purged_samples":purged_samples,"test_samples":len(test),
                                    "train_positives":int(ytrain.sum()),"test_positives":int(ytest.sum()),
                                    "calibration_layer_fitted":bool(calibration_fitted),
                                    "adaptation_hours":ADAPTATION_HOURS,"pretest_data_eligible":bool(qualifies),
                                    "event_rich_descriptive_only":bool(event_rich),
                                    "population_outer_fold":int(fold)})
        for strategy,score in outcomes.items():
            for idx,(_,row) in enumerate(test.iterrows()):
                prediction_rows.append({"participant_id":str(pid),"timestamp":row.timestamp,"label":int(row.label),
                                        "target_glucose":float(row.target_glucose),"strategy":strategy,
                                        "score":float(score[idx]),"regression":float(reg_outcomes[strategy][idx]),
                                        "pretest_data_eligible":bool(qualifies),
                                        "event_rich_descriptive_only":bool(event_rich)})
        print(f"chronological participant {pid}: {len(train)} early / {len(test)} later; eligible by pre-test data={bool(qualifies)}",flush=True)

    preds=pd.DataFrame(prediction_rows)
    preds.to_csv(OUT/"within_person_predictions_local.csv",index=False)
    ppl=pd.DataFrame(participant_results)
    ppl.to_csv(OUT/"within_person_participant_counts_local.csv",index=False)
    eligible=set(ppl.loc[ppl.pretest_data_eligible,"participant_id"])
    cls=preds.loc[preds.participant_id.isin(eligible)].copy()
    event_rich=set(ppl.loc[ppl.event_rich_descriptive_only,"participant_id"])
    cls_event_rich=preds.loc[preds.participant_id.isin(event_rich)].copy()
    regression={};classification={}
    for name,part in preds.groupby("strategy",sort=False):
        if name=="population_plus_early_calibration":
            continue
        regression[name]={"participants":int(part.participant_id.nunique()),
                          "pooled":regression_metrics(part.target_glucose,part.regression),
                          "equal_participant_macro":macro_regression(part,"regression")}
    for name,part in cls.groupby("strategy",sort=False):
        classification[name]={"eligible_participants":int(part.participant_id.nunique()),
                              "participants_with_both_test_classes":int(part.groupby("participant_id").label.nunique().eq(2).sum()),
                              "pooled":class_metrics(part.label,part.score),
                              "equal_participant_macro":participant_macro(part,"score")}
    event_rich_classification={}
    for name,part in cls_event_rich.groupby("strategy",sort=False):
        event_rich_classification[name]={"descriptive_only_participants":int(part.participant_id.nunique()),
                                         "participants_with_both_test_classes":int(part.groupby("participant_id").label.nunique().eq(2).sum()),
                                         "pooled":class_metrics(part.label,part.score),
                                         "equal_participant_macro":participant_macro(part,"score")}
    summary={"design":{"split":f"Chronological within each participant with a purged boundary: adaptation/training uses forecast rows whose exact t+{LABEL_HORIZON_MINUTES} label is observed by the {ADAPTATION_HOURS}-hour cutoff; rows forecast before the cutoff whose label arrives after it are purged; test uses rows forecast at or after the cutoff.",
                        "label_horizon_minutes":LABEL_HORIZON_MINUTES,"purged_samples_total":int(sum(x["purged_samples"] for x in participant_results)),
                        "primary_eligibility_rule":f"At least {MIN_EARLY_INTERVALS} usable labeled forecast intervals in the fixed, label-purged early window and at least one later test interval. Rule uses timestamps and early data availability only, not held-out labels.",
                        "eligible_participants":len(eligible),"all_participants":int(ppl.participant_id.nunique()),
                        "all_participant_test_samples":int(sum(x["test_samples"] for x in participant_results)),
                        "event_rich_secondary_rule":"Descriptive subgroup only: at least 10 positive labels in both adaptation and test segments. Future labels are not used for primary inclusion or headline results.",
                        "event_rich_descriptive_participants":len(event_rich),
                        "hyperparameters":"For each participant, use settings selected by grouped CV in a fold where that entire participant was held out; the participant's own timeline is excluded from its hyperparameter selection."},
             "classification_on_eligible_participants":classification,
             "classification_event_rich_descriptive_subgroup":event_rich_classification,
             "regression_on_all_participants":regression,
             "participant_counts_summary":{"participants":int(len(ppl)),"eligible_by_pretest_data":int(len(eligible)),
                 "train_positive_min":int(ppl.train_positives.min()),"test_positive_min":int(ppl.test_positives.min()),
                 "minimum_early_intervals_for_inclusion":MIN_EARLY_INTERVALS,
                 "adaptation_window_hours":ADAPTATION_HOURS,
                 "eligible_with_zero_test_events":int(ppl.loc[ppl.pretest_data_eligible,"test_positives"].eq(0).sum()),
                 "event_rich_descriptive_only":int(len(event_rich)),
                 "early_calibration_layer_fit_participants":int(calibration_fit_participants),
                 "calibration_layer_rule":"Regularized 1-D logistic calibration of the population model's early-window scores; fit only when the first 72 hours have both classes and at least five positive windows, otherwise fall back to population scores. This fallback does not affect primary eligibility.",
                 "median_early_interval_count":float(ppl.train_samples.median())}}
    (OUT/"within_person_metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    # Aggregate-only public artifacts; per-participant counts and predictions remain local.
    aggregate=json.loads(AGG_PATH.read_text(encoding="utf-8"))
    aggregate["within_person_chronological_validation"]=summary
    AGG_PATH.write_text(json.dumps(aggregate,indent=2,allow_nan=False),encoding="utf-8")
    strategies=list(classification)
    save_svg(DOC_FIGURES/"within_person_comparison.svg","Chronological adaptation: held-out performance",
             {"pooled PR-AUC":[classification[x]["pooled"]["pr_auc"] for x in strategies],
              "participant macro PR-AUC":[classification[x]["equal_participant_macro"]["pr_auc"] for x in strategies]},
             ["population","early-data retrain","early calibration","person only"],"PR-AUC")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
