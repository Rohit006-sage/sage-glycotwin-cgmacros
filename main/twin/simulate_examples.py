"""Generate local-only five-participant ODE/CGM traces; outputs are git-ignored."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .glucose_ode import ModelParameters, State, sensitivity_prior, step

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data/raw/cgmacros/cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0/extracted/CGMacros"
BIO=RAW/"bio.csv"
OUT=ROOT/"data/processed/twin_step1/simulation_examples"
SUMMARY=ROOT/"docs/results/twin_step1_ode_sanity.json"

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bio=pd.read_csv(BIO); bio.columns=[str(c).strip() for c in bio.columns]; bio=bio.set_index("subject")
    candidates=[]
    for p in sorted(RAW.glob("CGMacros-*/CGMacros-*.csv")):
        pid=int(p.parent.name.split("-")[-1]); b=bio.loc[pid]
        fg=pd.to_numeric(b.get("Fasting GLU - PDL (Lab)"),errors="coerce")
        ins=pd.to_numeric(b.get("Insulin"),errors="coerce")
        if pd.notna(fg) and pd.notna(ins) and fg>0 and ins>0:
            candidates.append((pid,float(fg),float(ins),sensitivity_prior(float(fg),float(ins))))
    # Deterministically cover the cohort HOMA/sensitivity range.
    candidates.sort(key=lambda x:x[3])
    picks=[candidates[round(q*(len(candidates)-1))] for q in np.linspace(0,1,5)]
    index=[]
    for rank,(pid,fg,ins,si) in enumerate(picks,1):
        path=RAW/f"CGMacros-{pid:03d}"/f"CGMacros-{pid:03d}.csv"
        d=pd.read_csv(path); d.columns=[str(c).strip() for c in d.columns]
        d["Timestamp"]=pd.to_datetime(d["Timestamp"],errors="coerce")
        d=d.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        g=pd.to_numeric(d.get("Libre GL"),errors="coerce")
        valid=g.notna(); d=d.loc[valid].reset_index(drop=True); g=pd.to_numeric(d["Libre GL"],errors="coerce")
        p=ModelParameters(insulin_sensitivity=si)
        # Use the available fasting lab value as the person's basal glucose;
        # initialize the dynamic state with the first CGM point.
        init=float(g.iloc[0]); baseline=fg; state=State(init); twin=[]
        carbs=pd.to_numeric(d.get("Carbs",pd.Series(0,index=d.index)),errors="coerce").fillna(0).clip(lower=0)
        if "METs" in d:
            met=pd.to_numeric(d["METs"],errors="coerce").fillna(10)/10
            activity_series=np.clip((met-1)/4,0,2)
        elif "Calories (Activity)" in d:
            activity_series=np.clip(pd.to_numeric(d["Calories (Activity)"],errors="coerce").fillna(0)/2,0,2)
        elif "Steps" in d:
            activity_series=np.clip(pd.to_numeric(d["Steps"],errors="coerce").fillna(0)/100,0,2)
        else:
            activity_series=pd.Series(0.0,index=d.index) # Categorical Intensity has no common documented scale.
        for i,row in d.iterrows():
            meal=float(carbs.iloc[i])
            activity=float(activity_series.iloc[i])
            if i:
                delta=(row.Timestamp-d.loc[i-1,"Timestamp"]).total_seconds()/60
                # Break long gaps into minute steps; carry activity only to the observed row.
                n=max(1,int(round(delta)))
                for k in range(n): state=step(state,delta/n,baseline_mg_dl=baseline,
                                               activity=activity if k==n-1 else 0,p=p)
            if meal:
                state=State(state.glucose_mg_dl,state.gut_fast_g+meal,state.gut_slow_g,
                            state.insulin_action_mg_dl,state.absorbed_g)
            twin.append(state.glucose_mg_dl)
        times=d["Timestamp"]
        fig,ax=plt.subplots(figsize=(11,4.5))
        ax.plot(times,g,color="#1f77b4",lw=1.2,label="Observed Libre CGM")
        ax.plot(times,twin,color="#d95f02",lw=1.3,label="Unfitted ODE simulation")
        meal_idx=np.flatnonzero(carbs.to_numpy()>0)
        if len(meal_idx): ax.scatter(times.iloc[meal_idx],g.iloc[meal_idx],s=14,color="#2ca02c",label="Logged meal")
        ax.set(title=f"Participant {pid:03d} · HOMA-seeded sensitivity {si:.2f}",ylabel="Glucose (mg/dL)",xlabel="Study time")
        ax.legend(loc="upper right",ncol=3,fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(OUT/f"participant_{rank:02d}_local.png",dpi=130); plt.close(fig)
        observed=np.asarray(g,dtype=float); simulated=np.asarray(twin,dtype=float)
        index.append({"plot":f"participant_{rank:02d}_local.png","participant_id":pid,
          "meal_rows_with_positive_carbs":int((carbs>0).sum()),"observed_cgm_points":int(len(g)),
          "homa_seeded_relative_sensitivity":si,"observed_range_mg_dl":[float(g.min()),float(g.max())],
          "ode_range_mg_dl":[float(np.min(twin)),float(np.max(twin))],
          "mae_local_mg_dl":float(np.mean(np.abs(observed-simulated))),
          "rmse_local_mg_dl":float(np.sqrt(np.mean((observed-simulated)**2))),
          "purpose":"local inspection only; not fitted/evaluated"})
    (OUT/"index_local.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
    summary={"participants":len(index),"selection":"Five deterministic HOMA-seeded sensitivity quantiles",
      "status":"Unfitted initialization sanity comparison; not a held-out model evaluation",
      "mean_participant_mae_mg_dl":float(np.mean([x["mae_local_mg_dl"] for x in index])),
      "mean_participant_rmse_mg_dl":float(np.mean([x["rmse_local_mg_dl"] for x in index])),
      "participant_rmse_range_mg_dl":[float(min(x["rmse_local_mg_dl"] for x in index)),float(max(x["rmse_local_mg_dl"] for x in index))],
      "plot_location":"Git-ignored data/processed/twin_step1/simulation_examples/; participant traces are not published"}
    SUMMARY.parent.mkdir(parents=True,exist_ok=True)
    SUMMARY.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Saved {len(index)} local-only participant plots under {OUT}")

if __name__=="__main__": main()
