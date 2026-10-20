"""Run HOMA-seeded EKF forecasts on all participants; writes only aggregate public results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error

from .glucose_ode import ModelParameters, State, sensitivity_prior
from .kalman import ExtendedGlucoseFilter

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/processed/revised_full"
RAW=ROOT/"data/raw/cgmacros/cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0/extracted/CGMacros"
BIO=RAW/"bio.csv"
OUT=ROOT/"data/processed/twin"
RESULT=ROOT/"docs/results/twin_stage2_state_estimation.json"
HORIZONS=(30,60,120)

def metrics(frame,score):
    y=frame.label.to_numpy(int); s=frame[score].to_numpy(float); target=frame.target_glucose.to_numpy(float)
    out={"n":len(frame),"event_prevalence":float(np.mean(y)),"pooled":{},"equal_participant_macro":{}}
    for key,mask in [("pr_auc",y.sum()>0),("roc_auc",np.unique(y).size==2)]:
        out["pooled"][key]=float(average_precision_score(y,s)) if mask else None
    groups=[]
    for _,g in frame.groupby("participant_id"):
        gy=g.label.to_numpy(int); gs=g[score].to_numpy(float); row={}
        if gy.sum()>0: row["pr_auc"]=float(average_precision_score(gy,gs))
        if np.unique(gy).size==2: row["roc_auc"]=float(roc_auc_score(gy,gs))
        row["mae"]=float(mean_absolute_error(g.target_glucose,g[score+"_reg"]))
        row["rmse"]=float(np.sqrt(mean_squared_error(g.target_glucose,g[score+"_reg"])))
        groups.append(row)
    for metric in ("pr_auc","roc_auc"):
        vals=[x[metric] for x in groups if metric in x]
        out["equal_participant_macro"][metric]=float(np.mean(vals)) if vals else None
    out["regression"]={"pooled_mae_mg_dl":float(mean_absolute_error(target,frame[score+"_reg"])),
                        "pooled_rmse_mg_dl":float(np.sqrt(mean_squared_error(target,frame[score+"_reg"]))),
                        "macro_mae_mg_dl":float(np.mean([x["mae"] for x in groups])),
                        "macro_rmse_mg_dl":float(np.mean([x["rmse"] for x in groups]))}
    return out

def activity_normalization(d):
    if "METs" in d:
        # Data dictionary records METs in tenths (e.g. 10 = 1 MET).
        return np.clip((pd.to_numeric(d["METs"],errors="coerce").fillna(10)/10-1)/4,0,2)
    if "Calories (Activity)" in d:
        return np.clip(pd.to_numeric(d["Calories (Activity)"],errors="coerce").fillna(0)/2,0,2)
    if "Steps" in d:
        return np.clip(pd.to_numeric(d["Steps"],errors="coerce").fillna(0)/100,0,2)
    return pd.Series(0.0,index=d.index)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bio=pd.read_csv(BIO); bio.columns=[str(c).strip() for c in bio.columns]; bio=bio.set_index("subject")
    samples=pd.read_csv(DATA/"rich_120m_samples_local.csv",dtype={"participant_id":str},parse_dates=["timestamp"])
    samples["participant_id"]=samples.participant_id.map(lambda x:f"{int(x):03d}")
    forecast_rows=[]; participant_status=[]
    for path in sorted(RAW.glob("CGMacros-*/CGMacros-*.csv")):
        pid=f"{int(path.parent.name.split('-')[-1]):03d}"
        subset=samples.loc[samples.participant_id.eq(pid)].sort_values("timestamp")
        d=pd.read_csv(path); d.columns=[str(c).strip() for c in d.columns]
        d["Timestamp"]=pd.to_datetime(d.Timestamp,errors="coerce")
        d=d.dropna(subset=["Timestamp"]).sort_values("Timestamp").drop_duplicates("Timestamp",keep="last").set_index("Timestamp")
        glucose=pd.to_numeric(d["Libre GL"],errors="coerce") if "Libre GL" in d else pd.Series(np.nan,index=d.index)
        meals=pd.to_numeric(d["Carbs"],errors="coerce").fillna(0).clip(lower=0) if "Carbs" in d else pd.Series(0.0,index=d.index)
        activity=activity_normalization(d)
        b=bio.loc[int(pid)]
        fg=pd.to_numeric(b.get("Fasting GLU - PDL (Lab)"),errors="coerce")
        insulin=pd.to_numeric(b.get("Insulin"),errors="coerce")
        valid_g=glucose.dropna()
        if valid_g.empty: continue
        baseline=float(fg) if pd.notna(fg) and fg>0 else float(valid_g.median())
        si=sensitivity_prior(float(fg),float(insulin)) if pd.notna(fg) and pd.notna(insulin) and fg>0 and insulin>0 else 1.0
        first=float(valid_g.iloc[0]); filt=ExtendedGlucoseFilter(State(first),baseline,ModelParameters(insulin_sensitivity=si))
        wanted=set(subset.timestamp.tolist()); raw_times=set(d.index.tolist()); timeline=sorted(wanted|raw_times)
        last_activity=0.0; prev=None; forecasts=0
        for t in timeline:
            if prev is not None:
                dt=max((t-prev).total_seconds()/60,1e-6)
                filt.predict(dt,activity=last_activity)
            obs=float(glucose.loc[t]) if t in glucose.index and pd.notna(glucose.loc[t]) else None
            filt.update(obs)
            if t in meals.index and meals.loc[t]>0: filt.add_meal(float(meals.loc[t]))
            if t in activity.index and pd.notna(activity.loc[t]): last_activity=float(activity.loc[t])
            if t in wanted:
                fc=filt.forecast_multiple(HORIZONS,dt_min=5.0)
                st=filt.state
                rec={"participant_id":pid,"timestamp":t,"filtered_glucose_mg_dl":st.glucose_mg_dl,
                     "latent_insulin_action_mg_dl":st.insulin_action_mg_dl,"gut_fast_g":st.gut_fast_g,
                     "gut_slow_g":st.gut_slow_g,"insulin_sensitivity":si,"baseline_glucose_mg_dl":baseline,
                     "filter_glucose_sd_mg_dl":float(np.sqrt(max(filt.covariance[0,0],0)))}
                for h,v in fc.items():
                    for k,val in v.items(): rec[f"twin_{h}m_{k}"]=val
                forecast_rows.append(rec); forecasts+=1
            prev=t
        participant_status.append({"participant_id":pid,"forecast_samples":forecasts,"HOMA_seed_sensitivity":si})
        print(f"state-estimation participant {pid}: {forecasts} forecast origins",flush=True)
    pred=pd.DataFrame(forecast_rows)
    pred.to_csv(OUT/"state_estimation_predictions_local.csv",index=False)
    base=[]
    for h in HORIZONS:
        src=pd.read_csv(DATA/f"horizon_{h}m_oof_local.csv",dtype={"participant_id":str},parse_dates=["timestamp"])
        if h==120:
            src=pd.read_csv(DATA/"horizon_120m_regression_local.csv",dtype={"participant_id":str},parse_dates=["timestamp"])
        src["participant_id"]=src.participant_id.map(lambda x:f"{int(x):03d}")
        joined=src.merge(pred,on=["participant_id","timestamp"],how="inner",validate="one_to_one")
        joined["current_glucose_reg"]=samples.set_index(["participant_id","timestamp"]).reindex(pd.MultiIndex.from_frame(joined[["participant_id","timestamp"]])).glucose_now.to_numpy()
        joined["twin_reg"]=joined[f"twin_{h}m_mean_mg_dl"]
        # Forecast level is also the ranking score for the elevated-glucose target.
        joined["twin_score"]=joined[f"twin_{h}m_mean_mg_dl"]
        joined["current_glucose_score"]=joined["current_glucose_reg"]
        # attach baseline predictions for direct reference only; stage 4 will refit same-fold models.
        joined["baseline_ml_score"]=joined["score"]
        joined["baseline_ml_reg"]=joined.get("regression",joined.target_glucose)
        for score in ("twin","current_glucose","baseline_ml"):
            joined[f"{score}_reg"]=joined[score+"_reg"] if score!="twin" else joined["twin_reg"]
        joined.to_csv(OUT/f"state_estimation_{h}m_local.csv",index=False)
        base.append({"horizon_minutes":h,"twin_only":metrics(joined,"twin_score"),
                     "persistence":metrics(joined,"current_glucose_score"),
                     "prior_ML_reference_not_same_training_subset":metrics(joined,"baseline_ml_score")})
    summary={"method":"HOMA-IR seeded EKF; assimilates each observed Libre reading; forecasts use no future meals/activity",
      "state":"CGM glucose, fast and slow gut carbohydrate, latent endogenous insulin action, cumulative absorbed carbohydrate",
      "band":"95% state-estimation interval from EKF covariance; does not include parameter uncertainty",
      "participants":len(participant_status),"forecast_rows":len(pred),"horizons":base,
      "participant_counts_are_local_only":True,
      "interpretation":"Research prototype. Baseline ML references are historical OOF predictions and are not an identical post-adaptation subset; final direct model comparison is performed in the grouped hybrid stage."}
    RESULT.parent.mkdir(parents=True,exist_ok=True)
    RESULT.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"participants":len(participant_status),"rows":len(pred),"result":str(RESULT)},indent=2))

if __name__=="__main__": main()
