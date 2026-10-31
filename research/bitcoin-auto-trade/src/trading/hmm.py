"""Walk-forward classification using only observations available at each close."""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


REGIME_NAMES = {0: "bear", 1: "side", 2: "bull"}


@dataclass(frozen=True)
class HMMConfig:
    train_window: int = 365
    retrain_every: int = 30
    n_iter: int = 100
    n_restarts: int = 10
    random_state: int = 42
    decode_window: int = 61
    sticky_kappa: float = 0.0

    def __post_init__(self):
        if self.train_window < 60 or min(self.retrain_every, self.n_iter, self.n_restarts, self.decode_window) < 1:
            raise ValueError("Invalid HMM configuration")
        if self.decode_window > self.train_window + 1:
            raise ValueError("decode_window must not exceed train_window + 1")
        if not np.isfinite(self.sticky_kappa) or self.sticky_kappa < 0:
            raise ValueError("sticky_kappa must be finite and nonnegative")


class StickyGaussianHMM(GaussianHMM):
    """Use the MAP objective, rather than plain likelihood, to stop MAP EM.

    hmmlearn otherwise may stop on a decreasing likelihood even while the
    regularized objective is still improving. This is a finite 3-state HMM,
    not the nonparametric Sticky HDP-HMM.
    """

    def _compute_lower_bound(self, curr_logprob):
        weights = np.asarray(self.transmat_prior) - 1.0
        return curr_logprob + float(np.sum(weights * np.log(np.maximum(self.transmat_, 1e-300))))


class HMMLabeler:
    def __init__(self, config: HMMConfig = HMMConfig()):
        self.config = config
        self.audit: list[dict] = []
        self.normalization_audit: list[dict] = []
        self.record_predictive_scores = False
        self.prediction_audit: list[dict] = []
        self.on_fit = None
        self.record_probabilities = False
        self.probability_audit: list[dict] = []

    def fit_predict(self, features: pd.DataFrame, label_values) -> np.ndarray:
        x = features.to_numpy(dtype=float)
        y = np.asarray(label_values, dtype=float)
        if len(y) != len(x) or y.ndim != 1 or not features.index.is_unique or not features.index.is_monotonic_increasing:
            raise ValueError("Features need a sorted unique index and aligned label_values")
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        labels = np.full(len(x), -1, dtype=np.int8)
        self.audit = []  # Reusing an instance must not retain a future model.
        self.normalization_audit = []
        self.prediction_audit = []
        self.probability_audit = []
        model = scaler = label_map = None
        next_fit = None
        cfg = self.config
        with threadpool_limits(limits=1):
            for i in range(cfg.train_window, len(x)):
                if not valid[i - cfg.train_window:i].all() or not valid[i]:
                    model = None
                    next_fit = None
                    continue
                if model is None or i == next_fit:
                    scaler, model, label_map, audit = self._fit(x[i-cfg.train_window:i], y[i-cfg.train_window:i])
                    transformed = scaler.transform(x[i-cfg.train_window:i])
                    for j, feature in enumerate(features.columns):
                        self.normalization_audit.append(dict(
                            fit_at=i, train_start=i-cfg.train_window, train_end=i-1,
                            feature=feature, mean=float(scaler.mean_[j]),
                            std_population=float(np.sqrt(scaler.var_[j])), scale=float(scaler.scale_[j]),
                            normalized_mean=float(transformed[:, j].mean()),
                            normalized_std_population=float(transformed[:, j].std(ddof=0))))
                    next_fit = i + cfg.retrain_every
                    for row in audit:
                        row.update(fit_at=i, train_start=i-cfg.train_window, train_end=i-1,
                                   train_rows=cfg.train_window)
                    self.audit.extend(audit)
                    if self.on_fit is not None:
                        self.on_fit(i,audit)
                window = x[max(0, i-cfg.decode_window+1):i+1]
                # All rows are contiguous and finite; never compress across missing bars.
                transformed = scaler.transform(window)
                raw_state = int(model.predict(transformed)[-1])
                labels[i] = label_map[raw_state]
                if self.record_probabilities:
                    # Only the endpoint of a window ending TODAY is retained.
                    # At that endpoint the posterior has no future emissions.
                    probabilities = model.predict_proba(transformed)[-1]
                    row = dict(position=i)
                    for raw,regime in label_map.items():
                        row[REGIME_NAMES[regime]] = float(probabilities[raw])
                    self.probability_audit.append(row)
                if self.record_predictive_scores:
                    before = model.score(transformed[:-1]) if len(window)>1 else 0.
                    self.prediction_audit.append(dict(position=i,
                        log_predictive_density=float(model.score(transformed)-before)))
        return labels

    def _fit(self, raw, y):
        scaler = StandardScaler().fit(raw)
        x = scaler.transform(raw)
        best = None
        errors = []
        for restart in range(self.config.n_restarts):
            try:
                cls = StickyGaussianHMM if self.config.sticky_kappa else GaussianHMM
                model = cls(n_components=3, covariance_type="diag", n_iter=self.config.n_iter,
                            random_state=self.config.random_state+restart,
                            transmat_prior=1.0+self.config.sticky_kappa*np.eye(3))
                model.fit(x)
                score = float(model.score(x))
                objective = score
                if self.config.sticky_kappa:
                    objective += self.config.sticky_kappa*float(np.log(np.diag(model.transmat_)).sum())
                states = model.predict(x)
                counts = np.bincount(states, minlength=3)
                if not np.isfinite(score) or (counts == 0).any():
                    raise ValueError("Nonfinite likelihood or unobserved state")
                means = [float(y[states == k].mean()) for k in range(3)]
                if best is None or objective > best[0]:
                    best = (objective, score, model, counts, means, self.config.random_state+restart)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                errors.append(str(exc))
        if best is None:
            raise RuntimeError(f"No usable 3-state HMM fit: {errors}")
        objective, score, model, counts, means, seed = best
        mapping = {int(raw): rank for rank, raw in enumerate(np.argsort(means, kind="stable"))}
        history = list(model.monitor_.history)
        delta = history[-1]-history[-2] if len(history) > 1 else None
        audit = [dict(raw_state=k, regime=REGIME_NAMES[mapping[k]], mean_trailing_log_return=means[k],
                      count=int(counts[k]), likelihood=score, objective=objective,
                      self_transition=float(model.transmat_[k,k]),
                      model_mean_duration=float(1/(1-model.transmat_[k,k])) if model.transmat_[k,k]<1 else float("inf"),
                      seed=seed, iterations=model.monitor_.iter,
                      converged=bool(delta is not None and 0 <= delta < model.tol),
                      last_likelihood_change=delta, tied_state_means=len(set(means)) < 3,
                      **{"config": asdict(self.config)}) for k in range(3)]
        return scaler, model, mapping, audit
