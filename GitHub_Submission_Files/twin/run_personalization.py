"""Fold-safe first-72-hour MAP personalization and identifiability audit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .glucose_ode import ModelParameters,sensitivity_prior
from .personalization import fit_person_parameters
from .run_state_estimation import RAW,BIO,activity_normalization

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"data/processed/twin"
RESULT=ROOT/"docs/results/twin_stage3_personalization.json"
SEED=42

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bio=pd.read_csv(BIO);bio.columns=[str(c).strip() for c in bio.columns];bio=bio.set_index("subject")
    profiles={}
    for path in sorted(RAW.glob("CGMacros-*/CGMacros-*.csv")):
        pid=f"{int(path.parent.name.split('-')[-1]):03d}"
        d=pd.read_csv(path);d.columns=[str(c).strip() for c in d.columns]
        d.Timestamp=pd.to_datetime(d.Timestamp,errors="coerce");d=d.dropna(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")
        g=pd.to_numeric(d["Libre GL"],errors="coerce")
        carbs=pd.to_numeric(d.get("Carbs",pd.Series(0,index=d.index)),errors="coerce").fillna(0).clip(lower=0)
        activity=activity_normalization(d)
        p15=pd.DataFrame({"glucose":g,"carbs":carbs,"activity":activity}).resample("15min").agg({"glucose":"last","carbs":"sum","activity":"mean"})
        b=bio.loc[int(pid)];fg=pd.to_numeric(b.get("Fasting GLU - PDL (Lab)"),errors="coerce");ins=pd.to_numeric(b.get("Insulin"),errors="coerce")
        if not pd.notna(fg) or fg<=0: fg=float(g.dropna().median())
        si=sensitivity_prior(float(fg),float(ins)) if pd.notna(ins) and ins>0 else 1.0
        start=p15.index.min();early=p15.loc[p15.index<start+pd.Timedelta(hours=72)]
        profiles[pid]={"early":early,"baseline":float(fg),"homa_si":si,"start":start}
    # Reuse the established seed-42 participant assignment exactly.
    old=pd.read_csv(ROOT/"data/processed/revised_full/primary_120m_seed_42_oof_local.csv",dtype={"participant_id":str})
    fold_by_pid=old.groupby("participant_id").outer_fold.nunique()
    if not fold_by_pid.eq(1).all(): raise ValueError("Expected one outer fold per participant")
    fold_by_pid=old.groupby("participant_id").outer_fold.first().to_dict()
    pids=np.array(sorted(profiles));folds=[]
    for fold in sorted(set(fold_by_pid.values())):
        test_ids={pid for pid in pids if fold_by_pid.get(pid)==fold}
        train_ids=set(pids)-test_ids
        folds.append((int(fold),train_ids,test_ids))
    output=[]
    for fold,train_ids,test_ids in folds:
        candidate={}
        for pid in train_ids:
            row=profiles[pid]; base=ModelParameters(insulin_sensitivity=row["homa_si"])
            z=row["early"]
            candidate[pid]=fit_person_parameters(z.glucose,z.carbs,z.activity,baseline_mg_dl=row["baseline"],prior=base)
        stable_train=[candidate[p].params for p in train_ids if candidate[p].stable]
        pop_si=float(np.median([p.insulin_sensitivity for p in stable_train])) if stable_train else float(np.median([profiles[p]["homa_si"] for p in train_ids]))
        pop_gain=float(np.median([p.glucose_per_gram for p in stable_train])) if stable_train else 0.75
        median_homa=float(np.median([profiles[p]["homa_si"] for p in train_ids]))
        for pid in pids:
            row=profiles[pid]; ratio=row["homa_si"]/max(median_homa,1e-5)
            prior=ModelParameters(insulin_sensitivity=float(np.clip(pop_si*ratio,.25,2.5)),glucose_per_gram=pop_gain)
            z=row["early"]
            fit=fit_person_parameters(z.glucose,z.carbs,z.activity,baseline_mg_dl=row["baseline"],prior=prior)
            output.append({"fold":fold,"participant_id":pid,"is_outer_test":pid in test_ids,
               "fit_stable":fit.stable,"condition_number":fit.condition_number,
               "objective_rmse_mg_dl":fit.objective_rmse_mg_dl,"fallback_reason":fit.reason,
               "insulin_sensitivity":fit.params.insulin_sensitivity,
               "glucose_per_gram":fit.params.glucose_per_gram,
               "population_prior_si":pop_si,"population_prior_gain":pop_gain,
               "homa_seed_si":row["homa_si"],"adaptation_hours":72})
        print(f"personalization fold {fold}: {len(train_ids)} prior-training people, {len(test_ids)} held-out people",flush=True)
    result=pd.DataFrame(output);result.to_csv(OUT/"personalized_parameters_local.csv",index=False)
    test=result.loc[result.is_outer_test]
    aggregate={"split":"5-fold participant grouped; each target participant uses only its first 72 hours; population MAP prior fit using outer-training participants only",
      "seed":SEED,"participants":int(len(profiles)),"outer_test_fits":int(len(test)),
      "stable_test_fits":int(test.fit_stable.sum()),"fallback_test_fits":int((~test.fit_stable).sum()),
      "fallback_reasons_test_only":{str(k):int(v) for k,v in test.fallback_reason.value_counts().items()},
      "test_parameter_ranges":{"insulin_sensitivity":[float(test.insulin_sensitivity.min()),float(test.insulin_sensitivity.max())],
        "glucose_per_gram":[float(test.glucose_per_gram.min()),float(test.glucose_per_gram.max())]},
      "median_condition_number_test":float(test.condition_number.replace([np.inf,-np.inf],np.nan).median()),
      "median_early_fit_rmse_mg_dl_test":float(test.objective_rmse_mg_dl.median()),
      "identifiability":"Only insulin sensitivity and effective carbohydrate gain are fitted; remaining time constants are fixed. Fits at bounds or ill-conditioned fits fall back to training-derived population parameters adjusted by the person's HOMA-IR prior. No participant-level estimates are published."}
    RESULT.parent.mkdir(parents=True,exist_ok=True);RESULT.write_text(json.dumps(aggregate,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(aggregate,indent=2))
if __name__=="__main__":main()
