"""Regularized early-window MAP fitting for two identifiable ODE parameters."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from .glucose_ode import ModelParameters, State, step


@dataclass(frozen=True)
class PersonalFit:
    params: ModelParameters
    stable: bool
    condition_number: float
    objective_rmse_mg_dl: float
    reason: str


def _predict(glucose, carbs_g, activity, baseline, p):
    state=State(float(glucose[0])); predictions=[state.glucose_mg_dl]
    for i in range(1,len(glucose)):
        meal=float(max(carbs_g[i-1],0.0))
        if meal:
            state=State(state.glucose_mg_dl,state.gut_fast_g+meal,state.gut_slow_g,
                        state.insulin_action_mg_dl,state.absorbed_g)
        state=step(state,15.0,baseline_mg_dl=baseline,
                   activity=float(max(activity[i-1],0.0)),p=p)
        predictions.append(state.glucose_mg_dl)
    return np.asarray(predictions)


def fit_person_parameters(glucose_mg_dl, carbs_g, activity, *, baseline_mg_dl: float,
                           prior: ModelParameters, max_points: int=288) -> PersonalFit:
    """Fit SI and carb gain with MAP shrinkage; caller must provide early-only data.

    Inputs are 15-minute rows. Other ODE constants remain fixed. A fit is marked
    unstable when the scaled local information matrix is ill-conditioned, an
    estimate hits a bound, or insufficient usable early observations exist.
    """
    g=np.asarray(glucose_mg_dl,dtype=float); carbs=np.asarray(carbs_g,dtype=float)
    activity=np.asarray(activity,dtype=float)
    n=min(len(g),len(carbs),len(activity),max_points)
    g,carbs,activity=g[:n],carbs[:n],activity[:n]
    valid=np.isfinite(g)
    g=np.where(valid,g,np.nanmedian(g[valid]) if valid.any() else baseline_mg_dl)
    carbs=np.nan_to_num(carbs,nan=0.0); activity=np.nan_to_num(activity,nan=0.0)
    if n<24 or valid.sum()<20:
        return PersonalFit(prior,False,float("inf"),float("nan"),"fewer than 20 valid rows")
    base=np.array([prior.insulin_sensitivity,prior.glucose_per_gram])
    scales=np.array([0.60,0.50]) # log-scale prior SDs; regularization set a priori.
    lo=np.log([0.25,0.1]); hi=np.log([2.5,2.0])
    def residual(theta):
        si,gain=np.exp(theta)
        pp=ModelParameters(**{**prior.__dict__,"insulin_sensitivity":float(si),"glucose_per_gram":float(gain)})
        pred=_predict(g,carbs,activity,baseline_mg_dl,pp)
        data=(pred[valid]-g[valid])/15.0
        reg=(theta-np.log(base))/scales
        return np.r_[data,reg]
    try:
        fit=least_squares(residual,np.log(base),bounds=(lo,hi),max_nfev=40,
                          x_scale="jac",ftol=1e-5,xtol=1e-5,gtol=1e-5)
        singular=np.linalg.svd(fit.jac,compute_uv=False)
        condition=float(singular[0]/max(singular[-1],1e-12))
        theta=fit.x; bound_hit=bool(np.any(theta-lo<1e-3) or np.any(hi-theta<1e-3))
        stable=bool(fit.success and np.isfinite(condition) and condition<1e7 and not bound_hit)
        pp=ModelParameters(**{**prior.__dict__,"insulin_sensitivity":float(np.exp(theta[0])),
                             "glucose_per_gram":float(np.exp(theta[1]))}) if stable else prior
        pred=_predict(g,carbs,activity,baseline_mg_dl,pp)
        rmse=float(np.sqrt(np.mean((pred[valid]-g[valid])**2)))
        reason="fit stable" if stable else ("parameter at bound" if bound_hit else "ill-conditioned or optimizer did not converge")
        return PersonalFit(pp,stable,condition,rmse,reason)
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        return PersonalFit(prior,False,float("inf"),float("nan"),f"fit failed: {type(exc).__name__}")
