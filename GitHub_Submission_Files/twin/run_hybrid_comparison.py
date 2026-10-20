"""Causally generate personalized twin features and evaluate ML-only/twin/hybrid.

Outer folds reuse the established seed-42 participant assignment. For each
participant, the first 72 h are adaptation only; all scored rows are later.
Population MAP priors and supervised models are fit on outer-training people.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,roc_auc_score,mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold

from .glucose_ode import ModelParameters,State
from .kalman import ExtendedGlucoseFilter
from .run_state_estimation import RAW,BIO,activity_normalization
from revised_evaluation import make_clf,make_reg
from project_config import RANDOM_SEED,BOOTSTRAP_SEED

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/processed/revised_full"; OUT=ROOT/"data/processed/twin"
PARAMS_PATH=OUT/"personalized_parameters_local.csv"
RESULT=ROOT/"docs/results/twin_stage4_comparison.json"
FIG=ROOT/"docs/figures/twin_stage4_comparison.svg"
HORIZONS=(30,60,120)
GRID=[{"learning_rate":.03,"max_iter":200,"max_leaf_nodes":15,"min_samples_leaf":100,"l2_regularization":1.0},
      {"learning_rate":.07,"max_iter":400,"max_leaf_nodes":31,"min_samples_leaf":300,"l2_regularization":3.0}]
BOOTSTRAPS=1000

def _base_activity_params(row,si,gain):
    return ModelParameters(insulin_sensitivity=float(np.clip(si,.25,2.5)),glucose_per_gram=float(np.clip(gain,.1,2.0)))

def generate_fold_features(fold,rich,fold_params,start_times):
    selected=rich.loc[rich.participant_id.isin(fold_params.participant_id)].copy()
    selected=selected.loc[[t>=start_times[pid]+pd.Timedelta(hours=72) for pid,t in zip(selected.participant_id,selected.timestamp)]]
    output=[]
    for path in sorted(RAW.glob("CGMacros-*/CGMacros-*.csv")):
        pid=f"{int(path.parent.name.split('-')[-1]):03d}"
        sample=selected.loc[selected.participant_id.eq(pid)]
        if sample.empty: continue
        par=fold_params.loc[fold_params.participant_id.eq(pid)].iloc[0]
        p=_base_activity_params(None,par.insulin_sensitivity,par.glucose_per_gram)
        d=pd.read_csv(path);d.columns=[str(c).strip() for c in d.columns]
        d.Timestamp=pd.to_datetime(d.Timestamp,errors="coerce");d=d.dropna(subset=["Timestamp"]).sort_values("Timestamp").drop_duplicates("Timestamp",keep="last").set_index("Timestamp")
        g=pd.to_numeric(d.get("Libre GL"),errors="coerce")
        carbs=pd.to_numeric(d.get("Carbs",pd.Series(0,index=d.index)),errors="coerce").fillna(0).clip(lower=0)
        act=activity_normalization(d)
        bio=pd.read_csv(BIO);bio.columns=[str(c).strip() for c in bio.columns];b=bio.loc[bio.subject.eq(int(pid))].iloc[0]
        fg=pd.to_numeric(b.get("Fasting GLU - PDL (Lab)"),errors="coerce")
        baseline=float(fg) if pd.notna(fg) and fg>0 else float(g.dropna().median())
        first=float(g.dropna().iloc[0]);filt=ExtendedGlucoseFilter(State(first),baseline,p)
        wanted=set(sample.timestamp.tolist());timeline=sorted(wanted|set(d.index.tolist()))
        lastact=0.;prev=None
        for t in timeline:
            if prev is not None: filt.predict(max((t-prev).total_seconds()/60,1e-6),activity=lastact)
            obs=float(g.loc[t]) if t in g.index and pd.notna(g.loc[t]) else None
            filt.update(obs)
            if t in carbs.index and carbs.loc[t]>0: filt.add_meal(float(carbs.loc[t]))
            if t in act.index and pd.notna(act.loc[t]): lastact=float(act.loc[t])
            if t in wanted:
                fc=filt.forecast_multiple(HORIZONS,dt_min=15.)
                s=filt.state
                row={"participant_id":pid,"timestamp":t,"filtered_glucose":s.glucose_mg_dl,
                  "latent_insulin_action":s.insulin_action_mg_dl,"gut_fast_g":s.gut_fast_g,
                  "gut_slow_g":s.gut_slow_g,"insulin_sensitivity":p.insulin_sensitivity,
                  "glucose_per_gram":p.glucose_per_gram,"fit_stable":bool(par.fit_stable),
                  "forecast_band_sd":float(np.sqrt(max(filt.covariance[0,0],0)))}
                for h,v in fc.items():
                    for name,val in v.items(): row[f"twin_{h}m_{name}"]=val
                output.append(row)
        print(f"hybrid twin feature fold {fold}, participant {pid}: {len(sample)} post-adaptation origins",flush=True)
    return pd.DataFrame(output)

def fit_classifier(X,y,groups,seed):
    best=None;best_score=-np.inf
    n_groups=len(np.unique(groups));n=min(3,n_groups)
    splitter=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed)
    splits=list(splitter.split(X,y,groups))
    for params in GRID:
        vals=[]
        for tr,va in splits:
            if np.unique(y[tr]).size<2 or y[va].sum()==0: continue
            model=make_clf(params).fit(X.iloc[tr],y[tr]);vals.append(average_precision_score(y[va],model.predict_proba(X.iloc[va])[:,1]))
        score=float(np.mean(vals)) if vals else -np.inf
        if score>best_score:best,best_score=params,score
    if best is None: best=GRID[0]
    return best,float(best_score)

def fit_regressor(X,y,groups,seed):
    best=None;best_score=np.inf;n=min(3,len(np.unique(groups)))
    for params in GRID:
        vals=[]
        for tr,va in GroupKFold(n_splits=n).split(X,y,groups):
            model=make_reg(params).fit(X.iloc[tr],y[tr]);vals.append(mean_absolute_error(y[va],model.predict(X.iloc[va])))
        score=float(np.mean(vals))
        if score<best_score:best,best_score=params,score
    return best,best_score

def metrics(frame,score,n_boot=BOOTSTRAPS):
    y=frame.label.to_numpy(int);s=frame[score].to_numpy(float);truth=frame.target_glucose.to_numpy(float)
    reg_score=score+"_reg" if score+"_reg" in frame.columns else score.removesuffix("_score")+"_reg"
    pids=frame.participant_id.to_numpy(str);unique=np.unique(pids)
    groups={pid:np.flatnonzero(pids==pid) for pid in unique}
    def point_metrics(ix,w=None):
        yy=y[ix];ss=s[ix];tt=truth[ix];ww=w
        pooled={"pr_auc":float(average_precision_score(yy,ss,sample_weight=ww)),
                "roc_auc":float(roc_auc_score(yy,ss,sample_weight=ww)) if np.unique(yy[ww>0] if ww is not None else yy).size==2 else None}
        macro_pr=[];macro_roc=[];maes=[];rmses=[];weights=[]
        if ww is None: local_weights=np.ones(len(ix))
        else: local_weights=ww
        local=pd.DataFrame({"pid":pids[ix],"y":yy,"s":ss,"truth":tt,
                            "pred":frame.loc[frame.index[ix],reg_score].to_numpy(float),"w":local_weights})
        for pid,g in local.groupby("pid"):
            if g.w.sum()==0: continue
            boot_people=float(g.w.iloc[0]) if ww is not None else 1.0
            if g.y.sum()>0: macro_pr.append((average_precision_score(g.y,g.s,sample_weight=g.w),boot_people))
            if np.unique(g.y[g.w>0]).size==2: macro_roc.append((roc_auc_score(g.y,g.s,sample_weight=g.w),boot_people))
            e=g.pred-g.truth; maes.append((np.average(np.abs(e),weights=g.w),boot_people))
            rmses.append((np.sqrt(np.average(e**2,weights=g.w)),boot_people))
        macro={"pr_auc":float(np.average([x[0] for x in macro_pr],weights=[x[1] for x in macro_pr])) if macro_pr else None,
               "roc_auc":float(np.average([x[0] for x in macro_roc],weights=[x[1] for x in macro_roc])) if macro_roc else None}
        reg={"pooled_mae_mg_dl":float(mean_absolute_error(tt,frame.loc[frame.index[ix],reg_score],sample_weight=ww)),
             "pooled_rmse_mg_dl":float(np.sqrt(mean_squared_error(tt,frame.loc[frame.index[ix],reg_score],sample_weight=ww))),
             "macro_mae_mg_dl":float(np.average([x[0] for x in maes],weights=[x[1] for x in maes])),
             "macro_rmse_mg_dl":float(np.average([x[0] for x in rmses],weights=[x[1] for x in rmses]))}
        return pooled,macro,reg
    all_ix=np.arange(len(frame));point=point_metrics(all_ix)
    rng=np.random.default_rng(BOOTSTRAP_SEED);boots={"pooled":{},"equal_participant_macro":{},"regression":{}}
    samples={k:[] for k in ["pooled_pr_auc","pooled_roc_auc","macro_pr_auc","macro_roc_auc","mae","rmse","macro_mae","macro_rmse"]}
    for _ in range(n_boot):
        draw=rng.choice(unique,size=len(unique),replace=True)
        counts=pd.Series(draw).value_counts().to_dict()
        w=np.array([counts.get(pid,0) for pid in pids],dtype=float)
        p,m,r=point_metrics(all_ix,w)
        for k,v in [("pooled_pr_auc",p["pr_auc"]),("pooled_roc_auc",p["roc_auc"]),("macro_pr_auc",m["pr_auc"]),
                    ("macro_roc_auc",m["roc_auc"]),("mae",r["pooled_mae_mg_dl"]),("rmse",r["pooled_rmse_mg_dl"]),
                    ("macro_mae",r["macro_mae_mg_dl"]),("macro_rmse",r["macro_rmse_mg_dl"])]:
            if v is not None and np.isfinite(v): samples[k].append(v)
    def ci(vals): return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if vals else [None,None]
    boots={"pooled":{"pr_auc_95_ci":ci(samples["pooled_pr_auc"]),"roc_auc_95_ci":ci(samples["pooled_roc_auc"])},
           "equal_participant_macro":{"pr_auc_95_ci":ci(samples["macro_pr_auc"]),"roc_auc_95_ci":ci(samples["macro_roc_auc"])},
           "regression":{"mae_95_ci":ci(samples["mae"]),"rmse_95_ci":ci(samples["rmse"]),
                         "macro_mae_95_ci":ci(samples["macro_mae"]),"macro_rmse_95_ci":ci(samples["macro_rmse"])}}
    return {"n":len(frame),"participants":len(unique),"event_prevalence":float(y.mean()),
            "pooled":point[0],"equal_participant_macro":point[1],"regression":point[2],"participant_cluster_bootstrap_95_ci":boots}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rich=pd.read_csv(DATA/"rich_120m_samples_local.csv",dtype={"participant_id":str},parse_dates=["timestamp"])
    rich.participant_id=rich.participant_id.map(lambda x:f"{int(x):03d}")
    old=pd.read_csv(DATA/"primary_120m_seed_42_oof_local.csv",dtype={"participant_id":str})
    fold_by_pid=old.groupby("participant_id").outer_fold.first().to_dict()
    # participant start times define each 72-hour adaptation cutoff
    start_times={}
    for path in RAW.glob("CGMacros-*/CGMacros-*.csv"):
        pid=f"{int(path.parent.name.split('-')[-1]):03d}"
        d=pd.read_csv(path,usecols=["Timestamp"]);d.Timestamp=pd.to_datetime(d.Timestamp,errors="coerce")
        start_times[pid]=d.Timestamp.min()
    params=pd.read_csv(PARAMS_PATH,dtype={"participant_id":str})
    all_preds={h:[] for h in HORIZONS}; fold_metadata=[]
    for fold in sorted(set(fold_by_pid.values())):
        fold_params=params.loc[params.fold.eq(fold)].copy()
        cache_path=OUT/f"hybrid_features_fold_{fold}_local.csv"
        if cache_path.exists():
            fold_feats=pd.read_csv(cache_path,dtype={"participant_id":str},parse_dates=["timestamp"])
        else:
            fold_feats=generate_fold_features(fold,rich,fold_params,start_times)
            fold_feats.to_csv(cache_path,index=False)
        # Remove early adaptation period; training and testing both use later data only.
        base=rich.drop(columns=["label","target_glucose","horizon_minutes"],errors="ignore")
        base=base.loc[[t>=start_times[pid]+pd.Timedelta(hours=72) for pid,t in zip(base.participant_id,base.timestamp)]].copy()
        base=base.merge(fold_feats,on=["participant_id","timestamp"],how="inner",validate="one_to_one")
        train_ids={pid for pid,f in fold_by_pid.items() if f!=fold};test_ids={pid for pid,f in fold_by_pid.items() if f==fold}
        for h in HORIZONS:
            fname=f"horizon_{h}m_oof_local.csv" if h<120 else "horizon_120m_regression_local.csv"
            target=pd.read_csv(DATA/fname,dtype={"participant_id":str},parse_dates=["timestamp"])
            target.participant_id=target.participant_id.map(lambda x:f"{int(x):03d}")
            frame=base.merge(target[["participant_id","timestamp","label","target_glucose"]],on=["participant_id","timestamp"],how="inner",validate="one_to_one")
            num_cols=[c for c in rich.columns if c not in {"participant_id","timestamp","target_glucose","label","horizon_minutes"}]
            twin_cols=["filtered_glucose","latent_insulin_action","gut_fast_g","gut_slow_g","insulin_sensitivity","glucose_per_gram","fit_stable",
                       f"twin_{h}m_mean_mg_dl",f"twin_{h}m_lower_95_mg_dl",f"twin_{h}m_upper_95_mg_dl","forecast_band_sd"]
            frame=frame.dropna(subset=num_cols+twin_cols+["label","target_glucose"]).copy()
            train=frame.loc[frame.participant_id.isin(train_ids)].reset_index(drop=True)
            test=frame.loc[frame.participant_id.isin(test_ids)].reset_index(drop=True)
            ytr=train.label.to_numpy(int);groups=train.participant_id.to_numpy(str);yt=test.label.to_numpy(int)
            results=[]
            for model_name,cols in [("ml_only",num_cols),("hybrid",num_cols+twin_cols)]:
                Xtr=train[cols].astype(float);Xte=test[cols].astype(float)
                cp,inner_ap=fit_classifier(Xtr,ytr,groups,RANDOM_SEED+fold+h)
                rp,inner_mae=fit_regressor(Xtr,train.target_glucose.to_numpy(float),groups,RANDOM_SEED+fold+h)
                cm=make_clf(cp).fit(Xtr,ytr);rm=make_reg(rp).fit(Xtr,train.target_glucose.to_numpy(float))
                score=cm.predict_proba(Xte)[:,1];reg=rm.predict(Xte)
                test[f"{model_name}_score"]=score;test[f"{model_name}_reg"]=reg
                test[f"{model_name}_score_reg"]=reg
                test[f"twin_score"]=test[f"twin_{h}m_mean_mg_dl"]
                test[f"twin_score_reg"]=test[f"twin_{h}m_mean_mg_dl"]
                results.append({"name":model_name,"classifier_params":cp,"inner_oof_ap":inner_ap,
                                "regressor_params":rp,"inner_oof_mae_mg_dl":inner_mae})
            # The same outer held-out participants and timestamps score all three strategies.
            test["outer_fold"]=fold;test["horizon_minutes"]=h
            all_preds[h].append(test[["participant_id","timestamp","outer_fold","horizon_minutes","label","target_glucose",
               "twin_score","twin_score_reg","ml_only_score","ml_only_reg","hybrid_score","hybrid_reg",
               f"twin_{h}m_lower_95_mg_dl",f"twin_{h}m_upper_95_mg_dl","insulin_sensitivity","glucose_per_gram","fit_stable"]])
            fold_metadata.append({"fold":fold,"horizon":h,"train_participants":len(train_ids),"test_participants":len(test_ids),"train_rows":len(train),"test_rows":len(test),"tuning":results})
            print(f"hybrid evaluation fold {fold}, horizon {h}: train={len(train)}, test={len(test)}",flush=True)
    summaries=[]
    for h,parts in all_preds.items():
        frame=pd.concat(parts,ignore_index=True)
        frame.to_csv(OUT/f"hybrid_comparison_{h}m_local.csv",index=False)
        item={"horizon_minutes":h,"twin_only":metrics(frame,"twin_score"),
              "ml_only":metrics(frame,"ml_only_score"),"hybrid":metrics(frame,"hybrid_score"),
              "same_outer_folds":True,"scoring":"held-out participants; only samples after each participant's fixed first 72 hours"}
        summaries.append(item)
    result={"design":{"seed":RANDOM_SEED,"bootstrap_seed":BOOTSTRAP_SEED,"bootstrap_replicates":BOOTSTRAPS,
      "outer_folds":"same participant assignment as existing seed-42 OOF","adaptation":"first 72 h in every participant; predictions scored later only",
      "training":"grouped inner folds choose HGB parameters on training participants only for AP and MAE separately",
      "forecast_inputs":"EKF assimilates CGM through current timestamp; future meal/activity inputs are zero/unknown",
      "prior":"training participant early-window MAP candidates only; personal HOMA-IR adjusts SI prior",
      "metric_convention":"macro PR-AUC averages participants with positives; macro ROC-AUC participants with both classes; regression macro gives equal weight to all participants"},
      "horizons":summaries,"fold_tuning":fold_metadata,
      "chronological_split":"The same OOF predictions are the first-72-hour adaptation / later-test evaluation; each participant is in exactly one held-out outer fold.",
      "clinical_limit":"Research evaluation only; no medication or treatment-response modeling; no external or prospective validation."}
    RESULT.parent.mkdir(parents=True,exist_ok=True);RESULT.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    # Aggregate-only chart of pooled PR-AUC and paired confidence bands.
    fig=[]
    labels=["Twin","ML only","Hybrid"]
    for item in summaries:
        vals=[item[n]["pooled"]["pr_auc"] for n in ("twin_only","ml_only","hybrid")]
        fig.append((item["horizon_minutes"],vals))
    svg=["<svg xmlns='http://www.w3.org/2000/svg' width='820' height='360' viewBox='0 0 820 360'>",
      "<rect width='100%' height='100%' fill='white'/><text x='24' y='30' font-size='20' font-family='Arial'>Personalized twin, ML and hybrid · pooled PR-AUC</text>"]
    colors=["#4169a1","#58a55c","#d07a36"]
    for i,(h,vals) in enumerate(fig):
        x0=90+i*245
        svg.append(f"<text x='{x0+45}' y='335' font-family='Arial' font-size='14'>{h}-minute horizon</text>")
        for j,(label,val) in enumerate(zip(labels,vals)):
            height=val*230;y=300-height;x=x0+j*58
            svg.append(f"<rect x='{x}' y='{y:.1f}' width='38' height='{height:.1f}' fill='{colors[j]}'/><text x='{x}' y='{y-5:.1f}' font-family='Arial' font-size='11'>{val:.3f}</text>")
            svg.append(f"<text x='{x-4}' y='317' font-family='Arial' font-size='9'>{label}</text>")
    svg.append("</svg>");FIG.parent.mkdir(parents=True,exist_ok=True);FIG.write_text("\n".join(svg),encoding="utf-8")
    print(json.dumps({"result":str(RESULT),"figure":str(FIG)},indent=2))

if __name__=="__main__":main()
