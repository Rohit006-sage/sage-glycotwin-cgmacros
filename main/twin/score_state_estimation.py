"""Score cached HOMA/EKF forecasts against persistence and historical ML OOF."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,mean_absolute_error,mean_squared_error

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/processed/revised_full"
OUT=ROOT/"data/processed/twin"; RESULT=ROOT/"docs/results/twin_stage2_state_estimation.json"

def metrics(frame,score):
    y=frame.label.to_numpy(int); s=frame[score].to_numpy(float); target=frame.target_glucose.to_numpy(float)
    out={"n":len(frame),"event_prevalence":float(y.mean()),"pooled":{},"equal_participant_macro":{}}
    out["pooled"]["pr_auc"]=float(average_precision_score(y,s)) if y.sum()>0 else None
    out["pooled"]["roc_auc"]=float(roc_auc_score(y,s)) if np.unique(y).size==2 else None
    groups=[]
    for _,g in frame.groupby("participant_id"):
        gy=g.label.to_numpy(int); gs=g[score].to_numpy(float); d={}
        if gy.sum()>0: d["pr_auc"]=float(average_precision_score(gy,gs))
        if np.unique(gy).size==2: d["roc_auc"]=float(roc_auc_score(gy,gs))
        d["mae"]=float(mean_absolute_error(g.target_glucose,g[score+"_reg"]))
        d["rmse"]=float(np.sqrt(mean_squared_error(g.target_glucose,g[score+"_reg"])))
        groups.append(d)
    for m in ("pr_auc","roc_auc"):
        v=[d[m] for d in groups if m in d]; out["equal_participant_macro"][m]=float(np.mean(v)) if v else None
    out["regression"]={"pooled_mae_mg_dl":float(mean_absolute_error(target,frame[score+"_reg"])),
        "pooled_rmse_mg_dl":float(np.sqrt(mean_squared_error(target,frame[score+"_reg"]))),
        "macro_mae_mg_dl":float(np.mean([d["mae"] for d in groups])),
        "macro_rmse_mg_dl":float(np.mean([d["rmse"] for d in groups]))}
    return out

def main():
    pred=pd.read_csv(OUT/"state_estimation_predictions_local.csv",dtype={"participant_id":str},parse_dates=["timestamp"])
    samples=pd.read_csv(DATA/"rich_120m_samples_local.csv",dtype={"participant_id":str},parse_dates=["timestamp"])
    samples.participant_id=samples.participant_id.map(lambda x:f"{int(x):03d}")
    base=[]
    for h in (30,60,120):
        name=f"horizon_{h}m_oof_local.csv" if h<120 else "horizon_120m_regression_local.csv"
        src=pd.read_csv(DATA/name,dtype={"participant_id":str},parse_dates=["timestamp"])
        src.participant_id=src.participant_id.map(lambda x:f"{int(x):03d}")
        j=src.merge(pred,on=["participant_id","timestamp"],how="inner",validate="one_to_one")
        ix=samples.set_index(["participant_id","timestamp"]).glucose_now
        mi=pd.MultiIndex.from_frame(j[["participant_id","timestamp"]])
        j["current_glucose_score"]=ix.reindex(mi).to_numpy()
        j["current_glucose_score_reg"]=j.current_glucose_score
        j["twin_score"]=j[f"twin_{h}m_mean_mg_dl"]
        j["twin_score_reg"]=j.twin_score
        j["baseline_ml_score"]=j.score
        j["baseline_ml_score_reg"]=j.regression if "regression" in j else j.target_glucose
        j=j.dropna(subset=["label","target_glucose","current_glucose_score","twin_score",
                           "baseline_ml_score","current_glucose_score_reg","twin_score_reg",
                           "baseline_ml_score_reg"]).copy()
        j.to_csv(OUT/f"state_estimation_{h}m_local.csv",index=False)
        twin=metrics(j,"twin_score"); persistence=metrics(j,"current_glucose_score")
        twin["beats_persistence_regression_mae"]=twin["regression"]["pooled_mae_mg_dl"] < persistence["regression"]["pooled_mae_mg_dl"]
        twin["beats_persistence_regression_rmse"]=twin["regression"]["pooled_rmse_mg_dl"] < persistence["regression"]["pooled_rmse_mg_dl"]
        base.append({"horizon_minutes":h,"twin_only":twin,
          "persistence":persistence,
          "historical_ML_OOF_reference_not_same_training_subset":metrics(j,"baseline_ml_score")})
    result={"method":"HOMA-IR seeded EKF; assimilates each observed Libre reading; no future meal/activity input in forecasts",
      "state":"glucose, fast/slow gut carbohydrate, latent endogenous insulin action, cumulative absorbed carbohydrate",
      "forecast_band":"95% EKF state uncertainty only; excludes parameter uncertainty",
      "participants":int(pred.participant_id.nunique()),"forecast_rows":len(pred),"horizons":base,
      "interpretation":"Stage decision uses paired MAE/RMSE against persistence. Historical ML references include all timeline points and are not on the post-adaptation subset used by the final comparison."}
    RESULT.parent.mkdir(parents=True,exist_ok=True); RESULT.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
