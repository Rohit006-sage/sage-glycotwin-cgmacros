"""Extended Kalman state estimator and input-limited glucose forecasts."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .glucose_ode import ModelParameters, State, step


@dataclass
class ExtendedGlucoseFilter:
    state: State
    baseline_mg_dl: float
    params: ModelParameters
    covariance: np.ndarray | None = None
    measurement_sd_mg_dl: float = 12.0
    process_sd: tuple[float, ...] = (0.7, 0.20, 0.15, 0.02, 0.002)

    def __post_init__(self):
        if self.covariance is None:
            self.covariance=np.diag([25.0,16.0,9.0,1.0,0.01])
        else:
            self.covariance=np.asarray(self.covariance,dtype=float)
        if self.covariance.shape != (5,5): raise ValueError("covariance must be 5x5")
        self.measurement_variance=float(self.measurement_sd_mg_dl**2)

    def predict(self, dt_min: float, activity: float = 0.0, *, substep: bool=True) -> None:
        """Propagate state and covariance through known/past activity input."""
        n=max(1,int(np.ceil(dt_min))) if substep else 1
        h=dt_min/n
        for _ in range(n):
            x=self.state.as_array()
            self.state=step(self.state,h,baseline_mg_dl=self.baseline_mg_dl,
                            activity=activity,p=self.params)
            # First-order EKF transition Jacobian of the continuous dynamics.
            g,fast,slow,action,_=x; p=self.params
            j=np.zeros((5,5),dtype=float)
            j[0,0]=-p.glucose_clearance_min*(1+p.activity_uptake*p.activity_sensitivity*max(activity,0))
            j[0,2]=p.glucose_per_gram/p.gut_absorb_min
            j[0,3]=-p.insulin_sensitivity*p.insulin_feedback_min
            j[1,1]=-1/p.gut_transit_min
            j[2,1]=1/p.gut_transit_min; j[2,2]=-1/p.gut_absorb_min
            j[3,0]=p.insulin_feedback_min if g>self.baseline_mg_dl else 0.0
            j[3,3]=-1/p.insulin_decay_min
            j[4,2]=1/p.gut_absorb_min
            f=np.eye(5)+h*j
            q=np.diag(np.square(np.asarray(self.process_sd))*h)
            self.covariance=f@self.covariance@f.T+q
            self.covariance=(self.covariance+self.covariance.T)*0.5

    def update(self, glucose_mg_dl: float | None) -> None:
        """Assimilate a CGM measurement when present."""
        if glucose_mg_dl is None or not np.isfinite(glucose_mg_dl): return
        h=np.array([[1.,0.,0.,0.,0.]])
        x=self.state.as_array(); innovation=float(glucose_mg_dl-x[0])
        s=float((h@self.covariance@h.T)[0,0]+self.measurement_variance)
        k=self.covariance@h.T/s
        x=x+(k[:,0]*innovation)
        # Joseph form retains positive semidefiniteness under roundoff.
        ident=np.eye(5); kh=k@h
        self.covariance=(ident-kh)@self.covariance@(ident-kh).T + k*self.measurement_variance@k.T
        self.covariance=(self.covariance+self.covariance.T)*0.5
        x[0]=max(0.,x[0]); x[1:]=np.maximum(0.,x[1:])
        self.state=State.from_array(x)

    def add_meal(self, carbs_g: float) -> None:
        if carbs_g<0: raise ValueError("carbohydrates must be nonnegative")
        x=self.state.as_array(); x[1]+=carbs_g
        self.state=State.from_array(x)
        self.covariance[1,1]+=max(0.1,0.04*carbs_g)**2

    def forecast(self, horizon_min: int=120, *, dt_min: float=5.0) -> dict[str,float]:
        """Forecast with no unobserved future meal/activity; band is EKF state uncertainty."""
        n=int(np.ceil(horizon_min/dt_min)); h=horizon_min/n
        filt=ExtendedGlucoseFilter(self.state,self.baseline_mg_dl,self.params,
              self.covariance.copy(),self.measurement_sd_mg_dl,self.process_sd)
        for _ in range(n): filt.predict(h,activity=0.)
        mean=filt.state.glucose_mg_dl
        sd=float(np.sqrt(max(filt.covariance[0,0],0)))
        return {"mean_mg_dl":mean,"lower_95_mg_dl":max(0.,mean-1.96*sd),
                "upper_95_mg_dl":mean+1.96*sd,"state_sd_mg_dl":sd}

    def forecast_multiple(self, horizons=(30,60,120), *, dt_min: float=5.0) -> dict[int,dict[str,float]]:
        """One forward pass for multiple horizons; no future inputs are supplied."""
        max_h=max(horizons); n=int(np.ceil(max_h/dt_min)); h=max_h/n
        filt=ExtendedGlucoseFilter(self.state,self.baseline_mg_dl,self.params,
              self.covariance.copy(),self.measurement_sd_mg_dl,self.process_sd)
        result={}
        for i in range(1,n+1):
            # Forecast uses a 15-minute RK4 step and covariance linearization;
            # this is intentionally coarser than the 1-minute filter updates.
            filt.predict(h,activity=0.,substep=False)
            minute=i*h
            for horizon in horizons:
                if horizon not in result and minute+1e-8>=horizon:
                    mean=filt.state.glucose_mg_dl
                    sd=float(np.sqrt(max(filt.covariance[0,0],0)))
                    result[int(horizon)]={"mean_mg_dl":mean,"lower_95_mg_dl":max(0.,mean-1.96*sd),
                                          "upper_95_mg_dl":mean+1.96*sd,"state_sd_mg_dl":sd}
        return result
