"""Aggregate-only CGMacros audit for fields used by the twin prototype."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cgmacros/cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0/extracted/CGMacros"
BIO = RAW / "bio.csv"
LOCAL = ROOT / "data/processed/twin_step1"
PUBLIC = ROOT / "docs/results/twin_field_audit.json"

def main() -> None:
    bio = pd.read_csv(BIO)
    bio.columns = [str(c).strip() for c in bio.columns]
    bio = bio.set_index("subject")
    details=[]; all_intervals=[]; all_rows=[]
    cgm_counts=[]; hr_counts=[]; activity_counts=[]; stream_gaps={"Libre GL":[],"Dexcom GL":[]}
    macro_cols=["Carbs","Protein","Fat","Fiber","Calories"]
    for path in sorted(RAW.glob("CGMacros-*/CGMacros-*.csv")):
        pid=int(path.parent.name.split("-")[-1]); d=pd.read_csv(path)
        d.columns=[str(c).strip() for c in d.columns]
        d["Timestamp"]=pd.to_datetime(d["Timestamp"],errors="coerce")
        d=d.sort_values("Timestamp")
        meal_mask=d.get("Meal Type",pd.Series(index=d.index,dtype=object)).notna()
        if "Carbs" in d: meal_mask |= pd.to_numeric(d["Carbs"],errors="coerce").fillna(0)>0
        meals=d.loc[meal_mask].copy()
        carbs=pd.to_numeric(meals.get("Carbs"),errors="coerce")
        meal_times=meals["Timestamp"].dropna().sort_values()
        gaps=meal_times.diff().dt.total_seconds()/60
        overlapping=int((gaps<120).sum())
        nonmissing={c:int(pd.to_numeric(meals[c],errors="coerce").notna().sum()) for c in macro_cols if c in meals}
        usable=int(pd.to_numeric(meals.get("Carbs"),errors="coerce").gt(0).sum()) if "Carbs" in meals else 0
        complete=int(pd.concat([pd.to_numeric(meals.get(c),errors="coerce") for c in ["Carbs","Protein","Fat","Fiber"] if c in meals],axis=1).notna().all(axis=1).sum()) if meals.shape[0] else 0
        g=pd.to_numeric(d.get("Libre GL"),errors="coerce")
        times=d.loc[g.notna(),"Timestamp"].dropna()
        gaps_cgm=times.sort_values().diff().dt.total_seconds()/60
        all_intervals.extend(gaps_cgm.dropna().tolist())
        for sensor in stream_gaps:
            if sensor in d:
                sensor_times=d.loc[pd.to_numeric(d[sensor],errors="coerce").notna(),"Timestamp"].dropna().sort_values()
                stream_gaps[sensor].extend((sensor_times.diff().dt.total_seconds()/60).dropna().tolist())
        cgm_counts.append(int(g.notna().sum()))
        hr_counts.append(int(pd.to_numeric(d.get("HR"),errors="coerce").notna().sum()))
        activity_cols=[c for c in ["METs","Intensity","Calories (Activity)","Steps"] if c in d]
        activity_col=", ".join(activity_cols) if activity_cols else None
        activity_counts.append(int(pd.to_numeric(d[activity_cols[0]],errors="coerce").notna().sum()) if activity_cols else 0)
        b=bio.loc[pid] if pid in bio.index else pd.Series(dtype=float)
        details.append({"participant":pid,"meal_events":int(len(meals)),"overlapping_meal_pairs_lt_120m":overlapping,
            "positive_carbohydrate_meals":usable,"complete_carb_protein_fat_fiber":complete,
            "usable_macro_counts":nonmissing,
            "carbohydrate_grams_sum":float(carbs.fillna(0).sum()),
            "fasting_glucose_mg_dl":pd.to_numeric(b.get("Fasting GLU - PDL (Lab)"),errors="coerce"),
            "fasting_insulin_uU_ml":pd.to_numeric(b.get("Insulin"),errors="coerce"),
            "cgm_readings":int(g.notna().sum()),"heart_rate_readings":hr_counts[-1],
            "activity_field":activity_col,"activity_readings":activity_counts[-1],
            "activity_column_readings":{c:int(pd.to_numeric(d[c],errors="coerce").notna().sum()) for c in activity_cols}})
        all_rows.append(d)
    LOCAL.mkdir(parents=True,exist_ok=True); PUBLIC.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(details).to_csv(LOCAL/"participant_meal_audit.csv",index=False)
    # Aggregate only; do not publish participant-level or sample-level rows.
    meal_counts=np.array([r["meal_events"] for r in details])
    overlap_counts=np.array([r["overlapping_meal_pairs_lt_120m"] for r in details])
    macros={k:sum((r["meal_events"] if k=="Meal Type" else 0) for r in details) for k in []}
    intervals=np.asarray(all_intervals,dtype=float)
    payload={"dataset":"CGMacros v1.0.0; 45 participants; audit is descriptive",
      "field_availability":{"meal_timestamp":"Timestamp on participant minute-grid; meal events marked by Meal Type or positive Carbs",
      "meal_macros":"Carbs, Protein, Fat, Fiber, Calories; counts of nonmissing meal-event values are reported below",
      "activity":"participant files vary: METs (documented as MET x10), Intensity, Calories (Activity), or Steps",
      "heart_rate":"HR in bpm, sampled on minute grid where available",
      "cgm":"Libre GL and Dexcom GL in mg/dL; timestamps are randomly shifted study times",
      "bio":"Fasting GLU - PDL (Lab) mg/dL and Insulin mcU/mL at study start"},
      "participants":len(details),"meal_events_total":int(meal_counts.sum()),
      "meal_events_per_participant":{"median":float(np.median(meal_counts)),"min":int(meal_counts.min()),"max":int(meal_counts.max())},
      "pairs_of_meals_within_120_minutes_total":int(overlap_counts.sum()),
      "participants_with_any_pair_within_120_minutes":int((overlap_counts>0).sum()),
      "positive_carbohydrate_meals_total":int(sum(r["positive_carbohydrate_meals"] for r in details)),
      "complete_carb_protein_fat_fiber_meals_total":int(sum(r["complete_carb_protein_fat_fiber"] for r in details)),
      "nonmissing_macro_cells_by_field":{c:int(sum(pd.to_numeric(df.get(c),errors="coerce").notna().sum() for df in all_rows)) for c in macro_cols},
      "cgm_interval_minutes":{"all_libre_readings_median":float(np.median(intervals)),"all_libre_readings_p10":float(np.quantile(intervals,.1)),"all_libre_readings_p90":float(np.quantile(intervals,.9)),
       "by_sensor":{sensor:{"median":float(np.median(vals)),"p10":float(np.quantile(vals,.1)),"p90":float(np.quantile(vals,.9)),"n_intervals":len(vals)} for sensor,vals in stream_gaps.items() if vals}},
      "participants_with_fasting_glucose_and_insulin":int(sum(pd.notna(r["fasting_glucose_mg_dl"]) and pd.notna(r["fasting_insulin_uU_ml"]) for r in details)),
      "activity_field_participants":{c:int(sum(c in (r["activity_field"] or "") for r in details)) for c in ["METs","Intensity","Calories (Activity)","Steps"]},
      "privacy":"Per-participant meal counts are saved only under git-ignored data/processed/twin_step1/. No per-sample outputs are written here."}
    PUBLIC.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
