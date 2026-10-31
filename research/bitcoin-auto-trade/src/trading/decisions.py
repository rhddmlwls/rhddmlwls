"""Causal evidence thresholds on named state probabilities; no minimum hold rule."""
import numpy as np
import pandas as pd


def probability_hysteresis(probabilities,ratio):
    if not np.isfinite(ratio) or ratio<1:
        raise ValueError("ratio must be finite and >=1")
    p=probabilities[["bear","side","bull"]].to_numpy(float)
    labels=np.full(len(p),-1,dtype=np.int8)
    state=None
    audit=[]
    for i,row in enumerate(p):
        if not np.isfinite(row).all():
            if not np.isnan(row).all():
                raise ValueError("Probabilities need either 3 finite values or an unavailable row")
            state=None
            continue
        if (row<0).any() or not np.isclose(row.sum(),1.,atol=1e-8):
            raise ValueError("Invalid probability simplex")
        candidate=int(row.argmax())
        before=state
        changed=False
        if state is None:
            state=candidate
        elif candidate!=state and row[candidate]>ratio*row[state]:
            state=candidate
            changed=True
        labels[i]=state
        audit.append(dict(position=i,previous=-1 if before is None else before,candidate=candidate,
            previous_probability=None if before is None else float(row[before]),
            candidate_probability=float(row[candidate]),ratio=float(ratio),switched=changed,regime=state))
    return pd.Series(labels,index=probabilities.index,name="regime"),pd.DataFrame(audit)


def probability_moving_average(probabilities,window):
    """Label only complete trailing daily posterior windows, including today.

    This averages published endpoint evidence; it never smooths historical
    states using future observations or fills a missing probability window.
    Exact ties choose the lowest state index, as an explicit argmax policy.
    """
    if type(window) is not int or window<1:
        raise ValueError("smoothing window must be a positive integer day count")
    if not isinstance(probabilities,pd.DataFrame):
        raise ValueError("smoothing probabilities must be a DataFrame")
    index=probabilities.index
    if (not isinstance(index,pd.DatetimeIndex) or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing
            or not index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()):
        raise ValueError("smoothing probabilities need consecutive daily dates")
    if not {"bear","side","bull"}.issubset(probabilities):
        raise ValueError("smoothing probabilities require bear, side and bull columns")
    p=probabilities[["bear","side","bull"]].astype(float)
    values=p.to_numpy()
    known=np.isfinite(values).all(axis=1)
    missing=np.isnan(values).all(axis=1)
    if not (known|missing).all():
        raise ValueError("Probabilities need either 3 finite values or an unavailable row")
    if ((values[known]<0).any() or (values[known]>1).any()
            or not np.allclose(values[known].sum(axis=1),1.,atol=1e-8,rtol=0)):
        raise ValueError("Invalid probability simplex")
    averaged=p.rolling(window,min_periods=window).mean()
    means=averaged.to_numpy()
    available=np.isfinite(means).all(axis=1)
    labels=np.full(len(index),-1,dtype=np.int8)
    labels[available]=means[available].argmax(axis=1)
    ties=np.zeros(len(index),dtype=bool)
    ties[available]=(means[available]==means[available].max(axis=1,keepdims=True)).sum(axis=1)>1
    previous=np.r_[-1,labels[:-1]] if len(index) else np.array([],dtype=np.int8)
    audit=pd.DataFrame(dict(position=np.arange(len(index)),window=window,
        window_start=index-pd.Timedelta(days=window-1),window_end=index,
        available_at=index+pd.Timedelta("1D")-pd.Timedelta("1ns"),
        available=available,previous=previous,candidate=labels,
        mean_bear=means[:,0],mean_side=means[:,1],mean_bull=means[:,2],
        regime=labels,tie=ties,
        switched=available&(previous>=0)&(labels!=previous)))
    return pd.Series(labels,index=index,name="regime"),audit
