"""Reusable HMM/HSMM fits; safe numeric snapshots, never pickle execution."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .hmm import HMMLabeler
from .research_cache import (array_digest,atomic_npz,commit_manifest,file_digest,fingerprint,read_manifest)


class FitCacheMixin:
    family="hmm"

    def __init__(self,config,cache_dir,versions):
        super().__init__(config)
        self.cache_dir=Path(cache_dir)
        self.versions=versions
        self.fit_cache_hits=0
        self.fit_cache_misses=0
        self.record_probabilities=True

    def _fit(self,raw,y):
        files=[Path(__file__),Path(__file__).with_name("hmm.py")]
        if self.family=="hsmm_poisson":
            files.append(Path(__file__).with_name("hsmm.py"))
        key=fingerprint(dict(family=self.family,config=asdict(self.config),raw=array_digest(raw),y=array_digest(y),
            engine={p.name:file_digest(p) for p in files},versions=self.versions))
        folder=self.cache_dir/"fits"/key
        metadata=read_manifest(folder,key,["parameters.npz"])
        if metadata is not None:
            with np.load(folder/"parameters.npz",allow_pickle=False) as values:
                scaler=StandardScaler()
                scaler.mean_,scaler.var_,scaler.scale_=[values[name].copy() for name in ["mean","var","scale"]]
                scaler.n_features_in_=len(scaler.mean_)
                scaler.n_samples_seen_=len(raw)
                if self.family=="hmm":
                    model=GaussianHMM(n_components=3,covariance_type="diag",init_params="",params="")
                    model.means_=values["means"].copy()
                    model.covars_=values["variances"].copy()
                else:
                    from .hsmm import GaussianPoissonHSMM
                    model=GaussianPoissonHSMM(self.config)
                    model.means_=values["means"].copy()
                    model.variances_=values["variances"].copy()
                    model.lambdas_=values["lambdas"].copy()
                model.startprob_=values["start"].copy()
                model.transmat_=values["transition"].copy()
            self.fit_cache_hits+=1
            return scaler,model,{int(k):v for k,v in metadata["mapping"].items()},metadata["audit"]
        self.fit_cache_misses+=1
        scaler,model,mapping,audit=super()._fit(raw,y)
        arrays=dict(mean=scaler.mean_,var=scaler.var_,scale=scaler.scale_,means=model.means_,
            variances=model._covars_ if self.family=="hmm" else model.variances_,
            start=model.startprob_,transition=model.transmat_)
        if self.family=="hsmm_poisson":
            arrays["lambdas"]=model.lambdas_
        atomic_npz(folder/"parameters.npz",**arrays)
        # Naming/audit floats can include an infinite geometric expectation;
        # store this diagnostic as null without changing fitted parameters.
        clean=[]
        for row in audit:
            clean.append({k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()})
        commit_manifest(folder,key,["parameters.npz"],mapping=mapping,audit=clean)
        return scaler,model,mapping,audit


class CachedHMMLabeler(FitCacheMixin,HMMLabeler):
    pass


def make_cached_labeler(spec,seed,cache_dir,versions):
    values=dict(train_window=spec.train_window,retrain_every=spec.retrain_every,n_iter=spec.iterations,
        n_restarts=spec.restarts,random_state=seed,decode_window=spec.decode_window,sticky_kappa=spec.sticky_kappa)
    if spec.kind=="hmm":
        from .hmm import HMMConfig
        return CachedHMMLabeler(HMMConfig(**values),cache_dir,versions)
    from .hsmm import HSMMConfig,HSMMLabeler
    class CachedHSMMLabeler(FitCacheMixin,HSMMLabeler):
        family="hsmm_poisson"
    return CachedHSMMLabeler(HSMMConfig(**values),cache_dir,versions)
